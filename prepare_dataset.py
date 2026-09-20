"""download GSM8K: 7000 train + 473 eval + 1319 test"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
SEED = 42
TRAIN_SIZE = 7000
EVAL_SIZE = 473
PROMPT = "Solve the following math problem step by step.\nPut your final numerical answer inside \\boxed{{}}.\n\nProblem:\n{question}\n\nSolution:\n"


def convert(item, source, index):
    reasoning, answer = item["answer"].rsplit("####", 1)
    reasoning = re.sub(r"<<.*?>>", "", reasoning, flags=re.DOTALL).strip()
    answer = answer.strip()
    if not answer:
        raise ValueError("Empty GSM8K answer")
    question = item["question"].strip()
    return {"id": f"gsm8k-{source}-{index:05d}",
            "question": question,
            "prompt": PROMPT.format(question=question),
            "response": f"{reasoning}\nTherefore, the final answer is \\boxed{{{answer}}}.",
            "gold_answer": answer}


def prepare(dataset):
    if len(dataset["train"]) != TRAIN_SIZE + EVAL_SIZE:
        raise ValueError("Expected 7473 official GSM8K training examples")
    indices = list(range(len(dataset["train"])))
    random.Random(SEED).shuffle(indices)
    splits = {
        "train": [convert(dataset["train"][i], "train", i) for i in indices[:TRAIN_SIZE]],
        "eval": [convert(dataset["train"][i], "train", i) for i in indices[TRAIN_SIZE:]],
        "test": [convert(item, "test", i) for i, item in enumerate(dataset["test"])],
    }
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = DATASET_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} examples")


if __name__ == "__main__":
    from datasets import load_dataset
    prepare(load_dataset("openai/gsm8k", "main"))
