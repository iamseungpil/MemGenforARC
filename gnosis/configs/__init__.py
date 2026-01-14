"""
Gnosis Configuration Module.

Provides utilities for loading and managing Gnosis configurations.

Usage:
    from gnosis.configs import load_config, get_default_config

    # Load from YAML file
    config = load_config("configs/gnosis.yaml")

    # Get default configuration
    default = get_default_config()

    # Merge custom values
    config = load_config("configs/gnosis.yaml", overrides={"training.learning_rate": 5e-5})
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Optional JSON support (stdlib, always available)
import json


def get_default_config() -> Dict[str, Any]:
    """Return default Gnosis configuration.

    Returns:
        Dictionary with default configuration values.
    """
    return {
        # Model configuration
        "model": {
            "name": "Qwen/Qwen3-8B",
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        },

        # Gnosis head configuration
        "gnosis": {
            "d_att": 256,
            "d_hid": 256,
            "d_conf": 128,
            "d_fusion": 256,
            "attn_map_size": 32,
            "selected_layers": None,  # None = last 8 layers
            "top_k": 10,
            "pdrop": 0.1,
            "stop_threshold": 0.5,
        },

        # Training configuration
        "training": {
            "output_dir": "./gnosis_output",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 500,
            "fp16": False,
            "bf16": True,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "pos_weight": 1.0,
            "freeze_llm": True,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_auroc",
            "greater_is_better": True,
        },

        # Data configuration
        "data": {
            "train_path": "data/gnosis/Final/merged_balanced.parquet",
            "eval_path": None,
            "eval_split": 0.1,
            "task": "math",
            "max_length": 2048,
            "text_column": "completion",
            "label_column": "correctness_label",
        },
    }


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def _set_nested_value(config: Dict, key_path: str, value: Any) -> None:
    """Set a nested value in a dictionary using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated key path (e.g., "training.learning_rate").
        value: Value to set.
    """
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def _get_nested_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """Get a nested value from a dictionary using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated key path.
        default: Default value if key not found.

    Returns:
        Value at key path or default.
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    use_defaults: bool = True,
) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file (YAML or JSON).
            If None, returns default configuration.
        overrides: Dictionary of overrides using dot notation keys.
            Example: {"training.learning_rate": 5e-5}
        use_defaults: Whether to merge with default configuration.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config_path doesn't exist.
        ValueError: If file format is unsupported.
    """
    # Start with defaults if requested
    if use_defaults:
        config = get_default_config()
    else:
        config = {}

    # Load from file if provided
    if config_path is not None:
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                )
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)

        elif suffix == ".json":
            with open(config_path, "r") as f:
                file_config = json.load(f)

        else:
            raise ValueError(
                f"Unsupported config format: {suffix}. "
                "Use .yaml, .yml, or .json"
            )

        # Merge file config with defaults
        if file_config:
            config = _deep_merge(config, file_config)

    # Apply overrides
    if overrides:
        for key_path, value in overrides.items():
            _set_nested_value(config, key_path, value)

    return config


def save_config(
    config: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "yaml",
) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary.
        output_path: Output file path.
        format: Output format ("yaml" or "json").
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "yaml":
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML output")
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    elif format == "json":
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration values.

    Args:
        config: Configuration dictionary.

    Returns:
        True if valid.

    Raises:
        ValueError: If configuration is invalid.
    """
    # Required top-level keys
    required_keys = ["model", "gnosis", "training", "data"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    # Validate model config
    model = config.get("model", {})
    if not model.get("name"):
        raise ValueError("model.name is required")

    # Validate training config
    training = config.get("training", {})
    if training.get("learning_rate", 0) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if training.get("num_train_epochs", 0) <= 0:
        raise ValueError("training.num_train_epochs must be positive")
    if training.get("per_device_train_batch_size", 0) <= 0:
        raise ValueError("training.per_device_train_batch_size must be positive")

    # Validate gnosis config
    gnosis = config.get("gnosis", {})
    if gnosis.get("d_att", 0) <= 0:
        raise ValueError("gnosis.d_att must be positive")
    if gnosis.get("d_hid", 0) <= 0:
        raise ValueError("gnosis.d_hid must be positive")
    if gnosis.get("d_conf", 0) <= 0:
        raise ValueError("gnosis.d_conf must be positive")

    # Validate data config
    data = config.get("data", {})
    if not data.get("train_path"):
        raise ValueError("data.train_path is required")

    return True


class ConfigNamespace:
    """Namespace wrapper for configuration dictionary.

    Allows attribute-style access to config values.

    Usage:
        config = load_config("config.yaml")
        ns = ConfigNamespace(config)
        print(ns.training.learning_rate)
    """

    def __init__(self, config: Dict[str, Any]):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"ConfigNamespace({self.to_dict()})"


__all__ = [
    "load_config",
    "save_config",
    "get_default_config",
    "validate_config",
    "ConfigNamespace",
]
