"""Symptom2Disease evaluation for baseline (single Ollama) and multi-agent pipeline.

Example:
    uv run python scripts/symptom2d_eval.py --limit 20 --runner baseline
    uv run python scripts/symptom2d_eval.py --limit 5 --runner multiagent
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

try:
    from langchain_ollama import ChatOllama  # modern package
except ImportError:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

load_dotenv()

PROMPT_CLASSIFY = (
    "You are a clinical classifier. Given a patient description, choose the single most likely disease "
    "from the allowed labels. Respond with compact JSON only: {\"label\": <label>, \"probabilities\": {label: float, ...}, "
    "\"explanation\": short}. Probabilities must be non-negative and sum to 1."
)


def _ollama_kwargs(model: str, temperature: float):
    base = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:11434"
    )
    if base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/")[:-3]
    return {
        "model": model,
        "temperature": temperature,
        "base_url": base,
        "api_key": os.getenv("OPENAI_API_KEY", "ollama"),
    }


def safe_json_parse(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Lenient: quote bare labels made of word chars
    import re
    patched = re.sub(r'"label"\s*:\s*([A-Za-z0-9_\-]+)', r'"label": "\1"', text)
    if patched != text:
        try:
            return json.loads(patched)
        except Exception:
            pass
    try:
        import ast
        return ast.literal_eval(text)
    except Exception:
        return None


def load_dataset(path: Path, limit: Optional[int]) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit and idx >= limit:
                break
            rows.append({"label": row["label"], "text": row["text"]})
    return rows


def accuracy(records: Iterable[Dict]) -> float:
    total = 0
    correct = 0
    for r in records:
        total += 1
        correct += 1 if r["pred"] == r["gold"] else 0
    return correct / total if total else 0.0


def macro_f1(records: Iterable[Dict]) -> float:
    labels = sorted({r["gold"] for r in records} | {r["pred"] for r in records})
    per_label: List[float] = []
    for label in labels:
        tp = sum(1 for r in records if r["pred"] == label and r["gold"] == label)
        fp = sum(1 for r in records if r["pred"] == label and r["gold"] != label)
        fn = sum(1 for r in records if r["pred"] != label and r["gold"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
        per_label.append(f1)
    return sum(per_label) / len(per_label) if per_label else 0.0


def brier_score(records: Iterable[Dict], labels: List[str]) -> float:
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


class BaselineRunner:
    def __init__(self, model: str, temperature: float = 0.1, max_retries: int = 3):
        self.llm = ChatOllama(**_ollama_kwargs(model, temperature))
        self.max_retries = max_retries

    def predict(self, text: str, labels: List[str]) -> Dict:
        label_list = ", ".join(labels)
        prompt = (
            f"{PROMPT_CLASSIFY}\nAllowed labels: {label_list}\n\nPatient description:\n{text}\n\n"
            "Return JSON only."
        )
        last_error: Optional[str] = None
        for _ in range(self.max_retries):
            msg = self.llm.invoke(prompt)
            content = getattr(msg, "content", "") or ""
            parsed = safe_json_parse(content)
            if parsed and self._valid(parsed, labels):
                return self._postprocess(parsed, labels)
            last_error = content
        # Graceful fallback on refusal or malformed output: uniform probabilities and echo explanation
        uniform = 1.0 / len(labels)
        return {
            "label": labels[0],
            "probabilities": {k: uniform for k in labels},
            "explanation": f"Fallback due to unparseable model output: {last_error}",
        }

    def _valid(self, payload: Dict, labels: List[str]) -> bool:
        choice = payload.get("label")
        probs = payload.get("probabilities")
        if choice not in labels:
            return False
        if not isinstance(probs, dict):
            return False
        # Allow extra keys; we will drop them later
        return all(isinstance(v, (int, float)) for v in probs.values())

    def _postprocess(self, payload: Dict, labels: List[str]) -> Dict:
        probs = payload.get("probabilities", {})
        # Keep only known labels when normalizing
        filtered = {k: v for k, v in probs.items() if k in labels}
        norm = self._normalize_probs(filtered, labels)
        choice = payload.get("label")
        if choice not in norm:
            choice = max(norm, key=norm.get)
        return {
            "label": choice,
            "probabilities": norm,
            "explanation": payload.get("explanation", ""),
        }

    @staticmethod
    def _normalize_probs(probs: Dict[str, float], labels: List[str]) -> Dict[str, float]:
        clean: Dict[str, float] = {k: float(max(0.0, probs.get(k, 0.0))) for k in labels}
        total = sum(clean.values())
        if total <= 0:
            uniform = 1.0 / len(labels)
            return {k: uniform for k in labels}
        return {k: v / total for k, v in clean.items()}


class MultiAgentRunner:
    def __init__(self, model: str, selector_model: Optional[str] = None):
        os.environ.setdefault("OLLAMA_MODEL", model)
        from langchain_core.messages import HumanMessage
        from langgraph.types import RunnableConfig
        from clinical_assistant.langgraph_module.multi_agent.supervisor.supervisor import (
            graph as supervisor_graph,
            SupervisorState,
        )
        self._HumanMessage = HumanMessage
        self._RunnableConfig = RunnableConfig
        self._graph = supervisor_graph
        self._State = SupervisorState
        self._config = RunnableConfig(configurable={"thread_id": "symptom2d", "recursion_limit": 50})
        self.selector = BaselineRunner(selector_model or model, temperature=0.0)

    async def _run_graph(self, prompt: str):
        return await self._graph.ainvoke(
            input=self._State(messages=[self._HumanMessage(content=prompt)]),
            config=self._config,
        )

    def _extract_explanation(self, state) -> str:
        expl = None
        explanation = getattr(state, "explanation", None) if not isinstance(state, dict) else state.get("explanation")
        if explanation:
            if isinstance(explanation, dict):
                parts = [explanation.get("clinician_summary"), explanation.get("patient_summary")]
                expl = "\n".join([p for p in parts if p])
            else:
                expl = str(explanation)
        if not expl:
            messages = getattr(state, "messages", None) if not isinstance(state, dict) else state.get("messages")
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if content:
                    expl = content
        return expl or ""

    def predict(self, text: str, labels: List[str]) -> Dict:
        prompt = (
            "You are a clinical multi-agent system. Analyze the patient's description, gather evidence, "
            "and reason about the most likely disease. Provide reasoning only; no label in the conversation.\n\n"
            f"Patient description:\n{text}\n"
        )
        state = asyncio.run(self._run_graph(prompt))
        explanation = self._extract_explanation(state)
        selector_input = (
            f"Reasoning context:\n{explanation}\n\n"
            "Choose the single best disease label."
        )
        selection = self.selector.predict(selector_input, labels)
        return {
            "label": selection["label"],
            "probabilities": selection["probabilities"],
            "explanation": explanation or selection.get("explanation", ""),
        }


def evaluate(dataset: Path, model: str, runner_name: str, limit: Optional[int], output: Optional[Path]):
    rows = load_dataset(dataset, limit)
    labels = sorted({r["label"] for r in rows})
    print(f"Starting Symptom2Disease eval: runner={runner_name}, model={model}, n={len(rows)}, labels={len(labels)}")
    runner = MultiAgentRunner(model) if runner_name == "multiagent" else BaselineRunner(model)

    records: List[Dict] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        if idx == 1:
            print(f"Processing {idx}/{total}")
        else:
            print(f"Processing {idx}/{total}", end="\r", flush=True)
        result = runner.predict(row["text"], labels)
        records.append(
            {
                "text": row["text"],
                "gold": row["label"],
                "pred": result["label"],
                "probabilities": result["probabilities"],
                "explanation": result.get("explanation", ""),
            }
        )

    acc = accuracy(records)
    f1 = macro_f1(records)
    brier = brier_score(records, labels)
    summary = {
        "n": len(records),
        "labels": len(labels),
        "accuracy": acc,
        "macro_f1": f1,
        "brier": brier,
    }
    print()  # newline after progress
    print(json.dumps(summary, indent=2))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {output}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symptom2Disease evaluation")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/Symptom2Disease.csv"))
    parser.add_argument("--model", type=str, default=None, help="Ollama model name; defaults to $OLLAMA_MODEL or 'llama3.1:8b'")
    parser.add_argument("--runner", type=str, default="baseline", choices=["baseline", "multiagent"], help="baseline=single LLM; multiagent=clinical pipeline + selector")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples (0 for full set)")
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSONL predictions")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    model = args.model or os.environ.get("OLLAMA_MODEL") or "llama3.1:8b"
    limit = args.limit if args.limit and args.limit > 0 else None
    output = args.output or Path(f"scripts/outputs/symptom2d_{args.runner}.jsonl")
    evaluate(
        dataset=args.dataset,
        model=model,
        runner_name=args.runner,
        limit=limit,
        output=output,
    )


if __name__ == "__main__":
    main()
