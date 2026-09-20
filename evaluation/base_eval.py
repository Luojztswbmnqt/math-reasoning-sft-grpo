"""Baseline generation, answer parsing, comparison and accuracy calculation."""
import argparse
import json
import logging
import sys
import re
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B"
DATA_FILE = ROOT / "dataset/test.jsonl"
OUTPUT_DIR = ROOT / "output/baseline"
MAX_PROMPT_LENGTH = 512
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256
MAX_SAMPLES = None  # Set to 8 for a quick debugging run.
SEED = 42
FINAL_ANSWER_PATTERNS = (
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"final\s+answer\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE),
)
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_and_tokenizer(model_path):
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    if use_bf16:
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info("Loading Base tokenizer: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Loading Base model: %s | dtype=%s", model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()

    logger.info("Base model loaded")
    return model, tokenizer, model_path


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the contents of the last balanced ``\\boxed{...}``."""
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        content_start = match.end()
        depth = 1
        for index in range(content_start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:index].strip()
    return None


def extract_answer(text: str, allow_numeric_fallback: bool = True) -> Optional[str]:
    """Extract an answer using boxed, explicit-final, then last-number rules."""
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed

    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            candidate = matches[-1].strip().rstrip(".$")
            number_matches = NUMBER_PATTERN.findall(candidate)
            return number_matches[-1] if number_matches else candidate

    if allow_numeric_fallback:
        numbers = NUMBER_PATTERN.findall(text)
        if numbers:
            return numbers[-1]
    return None


def _to_fraction(value: str) -> Optional[Fraction]:
    cleaned = value.strip()
    cleaned = re.sub(r"^\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}$", r"\1/\2", cleaned)
    cleaned = re.sub(r"^\\text\{([^{}]+)\}$", r"\1", cleaned)
    cleaned = cleaned.replace("$", "").replace(",", "")
    cleaned = cleaned.replace("\\%", "%")
    cleaned = cleaned.rstrip(". ")

    # A percentage is interpreted numerically, e.g. 25% == 25 for GSM8K's
    # answer convention. This strips formatting without dividing by 100.
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?", cleaned):
        numerator, denominator = (part.strip() for part in cleaned.split("/", maxsplit=1))
        try:
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        except (InvalidOperation, ZeroDivisionError, ValueError):
            return None

    try:
        return Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError):
        return None


def normalize_text_answer(value: str) -> str:
    """Normalize non-numeric fallback answers conservatively."""
    value = value.strip().lower()
    value = value.replace("$", "").replace(",", "")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(". ")


def answers_equal(predicted: Optional[str], gold: str) -> bool:
    """Compare numeric answers exactly, with a conservative text fallback."""
    if predicted is None:
        return False
    predicted_number = _to_fraction(predicted)
    gold_number = _to_fraction(gold)
    if predicted_number is not None and gold_number is not None:
        return predicted_number == gold_number
    return normalize_text_answer(predicted) == normalize_text_answer(gold)


@torch.inference_mode()
def evaluate(args: argparse.Namespace, model_loader=None) -> dict:
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("Batch size and generation length must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("Sample limit must be positive")

    set_seed(args.seed)
    logger.info("Loading evaluation dataset: %s", args.data_path)
    rows = read_jsonl(args.data_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("Evaluation dataset is empty. Run python prepare_dataset.py first.")

    logger.info("Dataset loaded | examples=%d", len(rows))
    if model_loader is None:
        model_loader = load_model_and_tokenizer
    model, tokenizer, model_label = model_loader(args.model_path)
    predictions: list[dict] = []
    start_time = time.perf_counter()

    logger.info(
        "Evaluation started | stage=%s | batch_size=%d | max_new_tokens=%d | decoding=greedy",
        args.stage,
        args.batch_size,
        args.max_new_tokens,
    )
    progress = tqdm(range(0, len(rows), args.batch_size), desc=f"Evaluating {args.stage}")
    running_correct = 0

    for batch_start in progress:
        batch = rows[batch_start : batch_start + args.batch_size]
        prompts = [row["prompt"] for row in batch]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
            return_tensors="pt",
        )
        for key, value in encoded.items():
            encoded[key] = value.to(model.device)
        input_width = encoded["input_ids"].shape[1]

        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        completion_ids = generated[:, input_width:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        for row, token_ids, completion in zip(batch, completion_ids, completions):
            predicted = extract_answer(completion, allow_numeric_fallback=True)
            boxed = extract_last_boxed(completion)
            token_list = token_ids.tolist()
            ended_with_eos = tokenizer.eos_token_id in token_list
            completion_length = (
                token_list.index(tokenizer.eos_token_id) + 1 if ended_with_eos else len(token_list)
            )
            correct = answers_equal(predicted, row["gold_answer"])
            running_correct += int(correct)
            predictions.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "predicted_answer": predicted,
                    "boxed_answer": boxed,
                    "correct": correct,
                    "has_valid_boxed_answer": boxed is not None,
                    "ended_with_eos": ended_with_eos,
                    "completion_tokens": completion_length,
                    "completion": completion,
                }
            )

        progress.set_postfix(
            accuracy=f"{running_correct / len(predictions):.2%}",
            correct=running_correct,
            evaluated=len(predictions),
        )

    elapsed = time.perf_counter() - start_time
    total = len(predictions)
    correct_count = sum(item["correct"] for item in predictions)
    boxed_count = sum(item["has_valid_boxed_answer"] for item in predictions)
    strict_boxed_correct = sum(
        item["correct"] and item["has_valid_boxed_answer"] for item in predictions
    )
    eos_count = sum(item["ended_with_eos"] for item in predictions)
    summary = {
        "stage": args.stage,
        "model": model_label,
        "base_model": args.model_path,
        "adapter_path": str(args.adapter_path) if getattr(args, "adapter_path", None) else None,
        "dataset": str(args.data_path),
        "examples": total,
        "correct": correct_count,
        "accuracy": correct_count / total,
        "boxed_format_count": boxed_count,
        "boxed_format_rate": boxed_count / total,
        "strict_boxed_accuracy": strict_boxed_correct / total,
        "ended_with_eos_count": eos_count,
        "possible_truncation_rate": 1.0 - eos_count / total,
        "mean_completion_tokens": sum(item["completion_tokens"] for item in predictions) / total,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_length": args.max_prompt_length,
        "batch_size": args.batch_size,
        "decoding": "greedy",
        "elapsed_seconds": elapsed,
        "examples_per_second": total / elapsed,
        "seed": args.seed,
    }

    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "incorrect_predictions.jsonl", (item for item in predictions if not item["correct"]))
    logger.info(
        "%s accuracy: %.2f%% (%d/%d) | boxed_format_rate=%.2f%% | possible_truncation_rate=%.2f%%",
        args.stage,
        summary["accuracy"] * 100,
        correct_count,
        total,
        summary["boxed_format_rate"] * 100,
        summary["possible_truncation_rate"] * 100,
    )
    logger.info("Evaluation results saved: %s", output_dir)
    return summary


def main():
    args = argparse.Namespace(
        stage="Base",
        model_path=MODEL,
        data_path=DATA_FILE,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )
    evaluate(args)


if __name__ == "__main__":
    main()
