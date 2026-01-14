# Dummy SymptomMultiAgentRunner for demonstration; replace with your actual multiagent logic
class SymptomMultiAgentRunner:
    def __init__(self, model: str, temperature: float = 0.1, max_retries: int = 3):
        from symptom2d_eval import BaselineRunner
        self.baseline = BaselineRunner(model, temperature, max_retries)

    def predict(self, text: str, labels: list[str]) -> dict:
        # Replace this with your actual multiagent logic for symptoms
        return self.baseline.predict(text, labels)
"""
Unified evaluation script for MedQA and Symptom tasks with baseline and multiagent runners.
Saves results incrementally after each evaluation for robustness.

Usage example:
    python scripts/unified_eval.py \
        --medqa-samples 20 --symptom-samples 20 \
        --output-dir scripts/outputs/

Arguments:
    --medqa-samples: Number of MedQA samples per runner
    --symptom-samples: Number of Symptom samples per runner
    --output-dir: Directory to save results
    --resume: Resume from existing results if present
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Optional

# Import runners from medqa_eval.py (assumes runners are in the same directory or PYTHONPATH)

# MedQA imports
from medqa_eval import (
    OllamaMCQRunner, MultiAgentRunner, load_examples, normalize_options, accuracy as medqa_accuracy, macro_f1 as medqa_macro_f1, brier_score as medqa_brier_score
)

# Symptom2Disease imports
import csv
from typing import Any

def load_symptom2d_dataset(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit and idx >= limit:
                break
            rows.append({"label": row["label"], "text": row["text"]})
    return rows

def symptom2d_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sorted({row["label"] for row in reader})

def symptom2d_accuracy(records):
    total = 0
    correct = 0
    for r in records:
        total += 1
        correct += 1 if r["pred"] == r["gold"] else 0
    return correct / total if total else 0.0

def symptom2d_macro_f1(records):
    labels = sorted({r["gold"] for r in records} | {r["pred"] for r in records})
    per_label = []
    for label in labels:
        tp = sum(1 for r in records if r["pred"] == label and r["gold"] == label)
        fp = sum(1 for r in records if r["pred"] == label and r["gold"] != label)
        fn = sum(1 for r in records if r["pred"] != label and r["gold"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
        per_label.append(f1)
    return sum(per_label) / len(per_label) if per_label else 0.0

def symptom2d_brier_score(records, labels):
    total = 0.0
    count = 0
    for r in records:
        probs = r["probabilities"]
        for label in labels:
            y = 1.0 if label == r["gold"] else 0.0
            p = probs.get(label, 0.0)
            total += (p - y) ** 2
        count += len(labels)
    return total / count if count else 0.0

def save_incremental(record: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_existing(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run_eval(task: str, runner_name: str, n_samples: int, output_path: Path, resume: bool):
    # Dataset root and region for each task
    if task == "medqa":
        dataset_root = Path("dataset/data_clean/questions")
        region = "US"
        split = "test"
        examples = load_examples(dataset_root, region, split, n_samples)
        get_gold = lambda ex: ex.get("answer_idx") or ex.get("answer")
        get_input = lambda ex: (ex["question"], normalize_options(ex["options"]))
        runner_map = {
            "baseline": OllamaMCQRunner("llama3.1:8b"),
            "multiagent": MultiAgentRunner("llama3.1:8b")
        }
    elif task == "symptom":
        dataset_root = Path("dataset/Symptom2Disease.csv")
        examples = load_symptom2d_dataset(dataset_root, n_samples)
        all_labels = symptom2d_labels(dataset_root)
        get_gold = lambda ex: ex["label"]
        get_input = lambda ex: (ex["text"], all_labels)
        # Use BaselineRunner and MultiAgentRunner for symptoms (assume same interface as medqa for now)
        from symptom2d_eval import BaselineRunner
        runner_map = {
            "baseline": BaselineRunner("llama3.1:8b"),
            "multiagent": SymptomMultiAgentRunner("llama3.1:8b")
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    # Load existing results if resuming (not used for metrics-only mode)
    existing = []
    done_questions = set()

    # Choose runner
    runner = MultiAgentRunner("llama3.1:8b") if runner_name == "multiagent" else OllamaMCQRunner("llama3.1:8b")

    saved = 0
    failed = 0
    total = min(n_samples, len(examples))
    all_records = []
    runner = runner_map[runner_name]
    for idx, ex in enumerate(examples, start=1):
        try:
            inp, opts = get_input(ex)
            gold = get_gold(ex)
            result = runner.predict(inp, opts)
            record = {
                "input": inp,
                "options": opts,
                "gold": gold,
                "pred": result.get("choice", result.get("label")),
                "probabilities": result["probabilities"],
                "confidence": result.get("confidence_override", result["probabilities"].get(result.get("choice", result.get("label")), 0.0)),
                "explanation": result.get("explanation", ""),
                "runner": runner_name,
                "task": task,
            }
            all_records.append(record)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[{task}/{runner_name}] Sample {idx} failed: {e}", flush=True)
        print(f"[{task}/{runner_name}] Progress: {saved}/{total} (failed: {failed})", end="\r", flush=True)
    print()  # Newline after progress
    if failed:
        print(f"[{task}/{runner_name}] {failed} samples failed and were skipped.")

    # Compute and save metrics after all samples are processed
    metrics = {}
    if task == "medqa":
        labels = sorted({k for k in all_records[0]["options"].keys()}) if all_records else []
        metrics = {
            "accuracy": medqa_accuracy(all_records),
            "macro_f1": medqa_macro_f1(all_records),
            "brier": medqa_brier_score(all_records, labels),
            "n": len(all_records),
        }
    elif task == "symptom":
        labels = sorted({k for k in all_records[0]["options"]}) if all_records else []
        metrics = {
            "accuracy": symptom2d_accuracy(all_records),
            "macro_f1": symptom2d_macro_f1(all_records),
            "brier": symptom2d_brier_score(all_records, labels),
            "n": len(all_records),
        }
    if metrics:
        metrics_path = output_path.with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[{task}/{runner_name}] Metrics saved to {metrics_path}")
        print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Unified MedQA/Symptom evaluation")
    parser.add_argument("--baseline-samples", type=int, default=10)
    parser.add_argument("--multiagent-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/outputs/"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # MedQA
    out_path = args.output_dir / "medqa_baseline.jsonl"
    metrics_path = out_path.with_suffix(".metrics.json")
    if not (args.resume and metrics_path.exists()):
        run_eval("medqa", "baseline", args.baseline_samples, out_path, args.resume)
    else:
        print(f"[medqa/baseline] Skipping: metrics already exist at {metrics_path}")

    out_path = args.output_dir / "medqa_multiagent.jsonl"
    metrics_path = out_path.with_suffix(".metrics.json")
    if not (args.resume and metrics_path.exists()):
        run_eval("medqa", "multiagent", args.multiagent_samples, out_path, args.resume)
    else:
        print(f"[medqa/multiagent] Skipping: metrics already exist at {metrics_path}")

    # Symptom2Disease
    out_path = args.output_dir / "symptom_baseline.jsonl"
    metrics_path = out_path.with_suffix(".metrics.json")
    if not (args.resume and metrics_path.exists()):
        run_eval("symptom", "baseline", args.baseline_samples, out_path, args.resume)
    else:
        print(f"[symptom/baseline] Skipping: metrics already exist at {metrics_path}")

    out_path = args.output_dir / "symptom_multiagent.jsonl"
    metrics_path = out_path.with_suffix(".metrics.json")
    if not (args.resume and metrics_path.exists()):
        run_eval("symptom", "multiagent", args.multiagent_samples, out_path, args.resume)
    else:
        print(f"[symptom/multiagent] Skipping: metrics already exist at {metrics_path}")

if __name__ == "__main__":
    main()
