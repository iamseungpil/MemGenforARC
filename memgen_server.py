"""
MemGen FastAPI Server for ARC Integration
OpenAI-compatible API with latent memory injection
"""
import argparse
import json
import logging
import re
import os
import threading
from typing import Optional
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import GenerationConfig

# MemGen imports
from memgen.model.modeling_memgen import MemGenModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MemGen Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global model reference with thread-safe loading
model: Optional[MemGenModel] = None
tokenizer = None
_model_lock = threading.Lock()


# Request/Response Models (OpenAI-compatible)
class Message(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str = "memgen-arc"
    messages: list[Message]
    max_tokens: int = 2048
    temperature: float = 0.0
    response_format: Optional[dict] = None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = "memgen-response"
    object: str = "chat.completion"
    model: str = "memgen-arc"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


def convert_messages_to_text(messages: list[Message]) -> list[dict]:
    """Convert OpenAI-format messages to simple chat format"""
    converted = []
    for msg in messages:
        content = msg.content
        # Handle structured content format from arc-lang
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in ["input_text", "output_text", "text"]:
                        text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))
            content = "\n".join(text_parts)
        converted.append({"role": msg.role, "content": content})
    return converted


def extract_json_from_response(text: str) -> str:
    """Extract JSON object from model response"""
    # Try to find JSON object
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
        r'\{.*?\}',  # Simple JSON
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in reversed(matches):  # Try last match first
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    return text


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint with MemGen latent memory"""
    global model, tokenizer

    if model is None:
        logger.error("Model not loaded - rejecting request")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert messages to chat format
        messages = convert_messages_to_text(request.messages)
        logger.debug(f"Processing request with {len(messages)} messages")

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Add JSON instruction if response_format is json_object
        if request.response_format and request.response_format.get("type") == "json_object":
            prompt = prompt.rstrip()
            if not prompt.endswith("```json"):
                prompt += "\n\nRespond with valid JSON only:\n```json\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Generate with MemGen (latent memory injection happens automatically)
        generation_config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature > 0,
            temperature=max(request.temperature, 0.01),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
        except RuntimeError as e:
            logger.error(f"Model generation failed: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise HTTPException(status_code=503, detail="GPU out of memory - try shorter input")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        # Decode response (skip input tokens)
        response_ids = output_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Clean up response for JSON
        if request.response_format and request.response_format.get("type") == "json_object":
            response_text = extract_json_from_response(response_text)

        logger.debug(f"Generated response: {len(response_text)} chars")

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=input_ids.shape[1],
                completion_tokens=len(response_ids),
                total_tokens=input_ids.shape[1] + len(response_ids)
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "memgen-arc", "object": "model", "owned_by": "local"}]
    }


def load_model(config_path: str):
    """Load MemGen model from config (thread-safe)"""
    global model, tokenizer

    with _model_lock:
        if model is not None:
            logger.info("Model already loaded, skipping reload")
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", config)

        logger.info(f"Loading MemGen model: {model_config.get('model_name')}")
        try:
            loaded_model = MemGenModel.from_config(model_config)
            loaded_model.eval()

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loaded_model = loaded_model.to(device)

            # Assign to globals only after successful load
            model = loaded_model
            tokenizer = model.tokenizer

            logger.info(f"Model loaded on {device}")
            logger.info(f"  - Max prompt augmentations: {model.config.max_prompt_aug_num}")
            logger.info(f"  - Max inference augmentations: {model.config.max_inference_aug_num}")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    # Load model
    load_model(args.config)

    # Start server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
