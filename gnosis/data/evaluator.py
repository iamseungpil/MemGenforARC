"""
Task-specific evaluators for completion correctness.

Based on evaluation methods from:
- Math: EleutherAI lm-evaluation-harness (hendrycks_math)
- Trivia: Standard text normalization and token matching
- MCQ: Letter extraction from boxed format

Each evaluator returns (is_correct: bool, is_parsed: bool) tuple.
"""

import re
import string
import unicodedata
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseEvaluator(ABC):
    """Abstract base class for task-specific evaluators."""

    @staticmethod
    @abstractmethod
    def evaluate(completion: str, gold_answer: str) -> Tuple[bool, bool]:
        """
        Evaluate a completion against the gold answer.

        Args:
            completion: Model's generated completion text.
            gold_answer: Ground truth answer.

        Returns:
            Tuple of (is_correct, is_parsed):
            - is_correct: Whether the extracted answer matches gold.
            - is_parsed: Whether an answer could be extracted from completion.
        """
        pass


class MathEvaluator(BaseEvaluator):
    """
    Evaluator for math problems.

    Extracts answer from \\boxed{} format and performs symbolic equivalence check.
    Adapted from EleutherAI lm-evaluation-harness.
    """

    @staticmethod
    def evaluate(completion: str, gold_answer: str) -> Tuple[bool, bool]:
        """
        Evaluate math completion.

        Args:
            completion: Model completion containing \\boxed{answer}.
            gold_answer: Ground truth answer (may or may not be in boxed format).

        Returns:
            (is_correct, is_parsed) tuple.
        """
        try:
            # Extract answer from completion
            pred_boxed = MathEvaluator._first_boxed_only_string(completion)
            if pred_boxed is None:
                return False, False

            pred_answer = MathEvaluator._remove_boxed(pred_boxed)
            if pred_answer is None:
                return False, False

            # Extract gold answer if it's in boxed format
            gold_boxed = MathEvaluator._last_boxed_only_string(gold_answer)
            if gold_boxed is not None:
                gold_clean = MathEvaluator._remove_boxed(gold_boxed)
            else:
                gold_clean = gold_answer.strip()

            # Check equivalence
            is_correct = MathEvaluator._is_equiv(pred_answer, gold_clean)
            return is_correct, True

        except Exception:
            return False, False

    @staticmethod
    def _first_boxed_only_string(text: str) -> Optional[str]:
        """Extract first \\boxed{...} from text."""
        if "\\boxed " in text:
            return "\\boxed " + text.split("\\boxed ")[1].split("$")[0]

        idx = text.find("\\boxed")
        if idx < 0:
            idx = text.find("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0

        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            if text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None
        return text[idx:right_brace_idx + 1]

    @staticmethod
    def _last_boxed_only_string(text: str) -> Optional[str]:
        """Extract last \\boxed{...} from text."""
        if "\\boxed " in text:
            return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]

        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0

        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            if text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None
        return text[idx:right_brace_idx + 1]

    @staticmethod
    def _remove_boxed(s: str) -> Optional[str]:
        """Remove \\boxed{} wrapper from string."""
        if s is None:
            return None

        if "\\boxed " in s:
            left = "\\boxed "
            if s.startswith(left):
                return s[len(left):]

        left = "\\boxed{"
        if s.startswith(left) and s.endswith("}"):
            return s[len(left):-1]

        # Handle \\fbox
        left = "\\fbox{"
        if s.startswith(left) and s.endswith("}"):
            return s[len(left):-1]

        return s

    @staticmethod
    def _is_equiv(str1: str, str2: str) -> bool:
        """Check if two math expressions are equivalent."""
        if str1 is None and str2 is None:
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = MathEvaluator._strip_string(str1)
            ss2 = MathEvaluator._strip_string(str2)
            return ss1 == ss2
        except Exception:
            return str1 == str2

    @staticmethod
    def _strip_string(text: str) -> str:
        """Normalize math string for comparison."""
        # Linebreaks
        text = text.replace("\n", "")

        # Remove inverse spaces
        text = text.replace("\\!", "")

        # Replace \\ with \
        text = text.replace("\\\\", "\\")

        # Replace tfrac and dfrac with frac
        text = text.replace("tfrac", "frac")
        text = text.replace("dfrac", "frac")

        # Remove \left and \right
        text = text.replace("\\left", "")
        text = text.replace("\\right", "")

        # Remove circ (degrees)
        text = text.replace("^{\\circ}", "")
        text = text.replace("^\\circ", "")

        # Remove dollar signs
        text = text.replace("\\$", "")

        # Remove units
        if "\\text{ " in text:
            splits = text.split("\\text{ ")
            if len(splits) == 2:
                text = splits[0]

        # Remove percentage
        text = text.replace("\\%", "")
        text = text.replace("%", "")

        # Handle decimal point at start
        text = text.replace(" .", " 0.")
        text = text.replace("{.", "{0.")
        if len(text) > 0 and text[0] == ".":
            text = "0" + text

        # Remove variable assignment (e.g., "k = ")
        if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
            text = text.split("=")[1]

        # Fix sqrt notation
        text = MathEvaluator._fix_sqrt(text)

        # Remove spaces
        text = text.replace(" ", "")

        # Fix fractions
        text = MathEvaluator._fix_fracs(text)

        # Manual conversion
        if text == "0.5":
            text = "\\frac{1}{2}"

        # Fix a/b notation
        text = MathEvaluator._fix_a_slash_b(text)

        return text

    @staticmethod
    def _fix_sqrt(text: str) -> str:
        """Fix sqrt notation (e.g., \\sqrt3 -> \\sqrt{3})."""
        if "\\sqrt" not in text:
            return text

        splits = text.split("\\sqrt")
        new_string = splits[0]

        for split in splits[1:]:
            if len(split) > 0 and split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr

        return new_string

    @staticmethod
    def _fix_fracs(text: str) -> str:
        """Fix fraction notation."""
        substrs = text.split("\\frac")
        new_str = substrs[0]

        if len(substrs) > 1:
            for substr in substrs[1:]:
                new_str += "\\frac"
                if len(substr) == 0 or substr[0] == "{":
                    new_str += substr
                else:
                    if len(substr) < 2:
                        return text
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            new_str += "{" + a + "}{" + b + "}" + substr[2:]
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            new_str += "{" + a + "}" + b + substr[2:]
                        else:
                            new_str += "{" + a + "}" + b

        return new_str

    @staticmethod
    def _fix_a_slash_b(text: str) -> str:
        """Convert a/b to \\frac{a}{b}."""
        if len(text.split("/")) != 2:
            return text

        a = text.split("/")[0]
        b = text.split("/")[1]

        try:
            a_int = int(a)
            b_int = int(b)
            if text == "{}/{}".format(a_int, b_int):
                return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
        except ValueError:
            pass

        return text


class TriviaEvaluator(BaseEvaluator):
    """
    Evaluator for trivia/QA tasks.

    Uses text normalization and token matching for comparison.
    """

    @staticmethod
    def evaluate(completion: str, gold_answer: str) -> Tuple[bool, bool]:
        """
        Evaluate trivia completion.

        Supports multiple gold answers separated by '|'.

        Args:
            completion: Model's generated answer.
            gold_answer: Ground truth answer(s), possibly separated by '|'.

        Returns:
            (is_correct, is_parsed) tuple.
        """
        try:
            # Extract answer - try boxed first, then take last sentence/phrase
            pred_answer = TriviaEvaluator._extract_answer(completion)
            if not pred_answer:
                return False, False

            # Normalize prediction
            pred_normalized = TriviaEvaluator._normalize_answer(pred_answer)

            # Handle multiple acceptable answers
            gold_answers = [a.strip() for a in gold_answer.split("|")]

            for gold in gold_answers:
                gold_normalized = TriviaEvaluator._normalize_answer(gold)

                # Exact match after normalization
                if pred_normalized == gold_normalized:
                    return True, True

                # Token-level F1 matching
                if TriviaEvaluator._token_match(pred_normalized, gold_normalized):
                    return True, True

            return False, True

        except Exception:
            return False, False

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract answer from completion text."""
        # Try boxed format first
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1)

        # Try "The answer is" pattern
        answer_match = re.search(
            r'(?:the answer is|answer:|final answer:?)\s*(.+?)(?:\.|$)',
            text.lower()
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Return last non-empty line
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]

        return text.strip()

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer text for comparison."""
        # Convert to lowercase
        text = text.lower()

        # Unicode normalization
        text = unicodedata.normalize("NFD", text)

        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    @staticmethod
    def _token_match(pred: str, gold: str, threshold: float = 0.5) -> bool:
        """
        Check if prediction tokens match gold tokens above threshold.

        Uses F1-style matching for partial credit.
        """
        pred_tokens = set(pred.split())
        gold_tokens = set(gold.split())

        if not gold_tokens:
            return not pred_tokens

        if not pred_tokens:
            return False

        # Calculate overlap
        common = pred_tokens & gold_tokens

        if not common:
            return False

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)

        if precision + recall == 0:
            return False

        f1 = 2 * precision * recall / (precision + recall)

        return f1 >= threshold


class MCQEvaluator(BaseEvaluator):
    """
    Evaluator for multiple choice questions.

    Extracts letter choice (A-F) from \\boxed{} format.
    """

    # Valid MCQ options
    VALID_OPTIONS = set("ABCDEF")

    @staticmethod
    def evaluate(completion: str, gold_answer: str) -> Tuple[bool, bool]:
        """
        Evaluate MCQ completion.

        Args:
            completion: Model completion with answer in \\boxed{X} format.
            gold_answer: Ground truth option letter (A-F).

        Returns:
            (is_correct, is_parsed) tuple.
        """
        try:
            # Extract predicted option
            pred_option = MCQEvaluator._extract_option(completion)
            if pred_option is None:
                return False, False

            # Normalize gold answer
            gold_option = gold_answer.strip().upper()
            if gold_option not in MCQEvaluator.VALID_OPTIONS:
                # Try to extract from gold if it's in a format like "(A)" or "A."
                gold_option = MCQEvaluator._extract_option(gold_answer)
                if gold_option is None:
                    return False, True  # Can't validate gold

            is_correct = pred_option == gold_option
            return is_correct, True

        except Exception:
            return False, False

    @staticmethod
    def _extract_option(text: str) -> Optional[str]:
        """Extract MCQ option letter from text."""
        # Try boxed format first
        boxed_match = re.search(r'\\boxed\{([A-Fa-f])\}', text)
        if boxed_match:
            return boxed_match.group(1).upper()

        # Try patterns like "The answer is (A)" or "Answer: A"
        patterns = [
            r'(?:answer is|answer:)\s*\(?([A-Fa-f])\)?',
            r'\(([A-Fa-f])\)\s*$',  # Ends with (A)
            r'^([A-Fa-f])\.?\s*$',  # Just the letter
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        # Fallback: look for standalone letters A-F in last line
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip().upper()
            for char in reversed(last_line):
                if char in MCQEvaluator.VALID_OPTIONS:
                    return char

        return None


def get_evaluator(task: str) -> type:
    """
    Get evaluator class for a given task.

    Args:
        task: One of 'math', 'trivia', 'mcq'.

    Returns:
        Evaluator class.

    Raises:
        ValueError: If task is not recognized.
    """
    evaluators = {
        "math": MathEvaluator,
        "trivia": TriviaEvaluator,
        "mcq": MCQEvaluator,
    }

    if task not in evaluators:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Supported tasks: {list(evaluators.keys())}"
        )

    return evaluators[task]
