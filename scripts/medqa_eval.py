"""MedQA evaluation harness with Ollama baseline and optional multi-agent pipeline,
plus LLM-based explanation judge.

Run example:
    python scripts/medqa_eval.py --region US --split test --limit 50 \
        --model llama3 --judge-model llama3
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
try:
    from langchain_ollama import ChatOllama  # modern package
except ImportError:  # fallback for LangChain community namespace
    from langchain_community.chat_models import ChatOllama

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Load local .env so OLLAMA_MODEL / OPENAI_BASE_URL are available when not exported.
load_dotenv()


PROMPT_MC = (
    "You are a careful medical exam solver. Read the question and options, then "
    "pick the single best option. Respond with compact JSON only. Schema: "
    "{\"choice\": one of [A,B,C,D,E], \"probabilities\": {A: float, ...}, "
    "\"explanation\": short string}. Probabilities must be non-negative and sum to 1.0."
)

PROMPT_JUDGE = (
    "You are an impartial medical evaluator. Score the provided explanation for "
    "consistency with the chosen answer and clinical plausibility. Return JSON: "
    "{\"plausibility\": number 0-1, \"consistency\": number 0-1, \"comment\": short}."
)


def load_examples(dataset_root: Path, region: str, split: str, limit: Optional[int]) -> List[Dict]:
    path = dataset_root / region / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    examples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            payload = json.loads(line)
            examples.append(payload)
    return examples


def normalize_options(options: Dict[str, str]) -> Dict[str, str]:
    ordered: Dict[str, str] = {}
    for key in sorted(options.keys()):
        ordered[key] = options[key]
    return ordered


def safe_json_parse(text: str) -> Optional[Dict]:
    # Clean possible markdown block markers
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    import re
    # Heuristics for common LLM malformed JSON
    patched = text
    
    # 1. Quote unquoted choice value: "choice": C -> "choice": "C"
    patched = re.sub(r'\"choice\"\s*:\s*([A-E])\b', r'"choice": "\1"', patched)

    # 2. Quote unquoted keys for options A-E:  A: 0.5 -> "A": 0.5
    # Look for A-E at start of line or after { or , followed by :
    patched = re.sub(r'(?<=[{,])\s*([A-E])\s*:', r'"\1":', patched)
    
    # 3. Handle case where keys are unquoted at start of lines (common in pretty printed JS objects)
    #    A: 0.1 
    patched = re.sub(r'^\s*([A-E])\s*:', r'"\1":', patched, flags=re.MULTILINE)

    if patched != text:
        try:
            return json.loads(patched)
        except Exception:
            pass

    # Last resort: literal_eval for permissive Python-literal-like outputs
    try:
        import ast
        return ast.literal_eval(text)
    except Exception:
        return None


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


class OllamaMCQRunner:
    def __init__(self, model: str, temperature: float = 0.1, max_retries: int = 3):
        self.llm = ChatOllama(**_ollama_kwargs(model, temperature))
        self.max_retries = max_retries

    def predict(self, question: str, options: Dict[str, str]) -> Dict:
        option_block = "\n".join([f"{k}. {v}" for k, v in options.items()])
        prompt = (
            f"{PROMPT_MC}\n\nQuestion:\n{question}\n\nOptions:\n{option_block}\n\n"
            "Return JSON only."
        )
        last_error: Optional[str] = None
        for _ in range(self.max_retries):
            msg = self.llm.invoke(prompt)
            content = getattr(msg, "content", "") or ""
            parsed = safe_json_parse(content)
            if parsed and self._valid(parsed, options):
                return self._postprocess(parsed, options)
            last_error = content
        raise ValueError(f"Failed to parse model output: {last_error}")

    def _valid(self, payload: Dict, options: Dict[str, str]) -> bool:
        choice = payload.get("choice")
        probs = payload.get("probabilities")
        if not isinstance(probs, dict):
            return False

        # Accept as long as there is at least one numeric prob (even if key not in options).
        has_numeric = any(isinstance(v, (int, float)) for v in probs.values())
        if not has_numeric:
            return False

        # Allow choice to be missing/invalid; fallback will pick argmax later.
        return True

    def _postprocess(self, payload: Dict, options: Dict[str, str]) -> Dict:
        probs = payload.get("probabilities", {})
        # Drop any keys not in the option set before normalizing.
        filtered = {k: v for k, v in probs.items() if k in options and isinstance(v, (int, float))}
        if not filtered:
            # If model produced only non-matching keys (e.g., I/II/III/IV), fall back to uniform.
            filtered = {k: 1.0 for k in options}
        norm = self._normalize_probs(filtered, options)
        choice = payload.get("choice")
        if choice not in norm:
            choice = max(norm, key=norm.get)
        return {
            "choice": choice,
            "probabilities": norm,
            "explanation": payload.get("explanation", ""),
        }

    @staticmethod
    def _normalize_probs(probs: Dict[str, float], options: Dict[str, str]) -> Dict[str, float]:
        clean: Dict[str, float] = {k: float(max(0.0, probs.get(k, 0.0))) for k in options}
        total = sum(clean.values())
        if total <= 0:
            uniform = 1.0 / len(options)
            return {k: uniform for k in options}
        return {k: v / total for k, v in clean.items()}


class MultiAgentRunner:
    """Wrap the clinical multi-agent supervisor and map its reasoning to MCQ outputs."""

    def __init__(self, model: str, selector_model: Optional[str] = None):
        # Ensure the pipeline uses the requested Ollama model by default.
        os.environ.setdefault("OLLAMA_MODEL", model)
        os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))

        # Lazy import to avoid loading graph when not needed.
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
        self._config = RunnableConfig(configurable={"thread_id": "eval", "recursion_limit": 50})
        # Selector converts pipeline reasoning to a MCQ choice.
        self.selector = OllamaMCQRunner(selector_model or model, temperature=0.0)

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

    def _extract_plan_confidence(self, state) -> Optional[float]:
        plan = getattr(state, "plan", None) if not isinstance(state, dict) else state.get("plan")
        if plan:
            if isinstance(plan, dict):
                return float(plan.get("overall_confidence", 0.0)) if "overall_confidence" in plan else None
            if hasattr(plan, "overall_confidence"):
                try:
                    return float(plan.overall_confidence)
                except Exception:
                    return None
        return None

    def predict(self, question: str, options: Dict[str, str]) -> Dict:
        prompt = (
            "You are a clinical multi-agent system solving a multiple-choice exam question. "
            "Analyze the case and options, gather evidence, validate hypotheses, and produce concise reasoning. "
            "Do not return an option letter in the conversation; just reason about the best option.\n\n"
            f"Question:\n{question}\n\n"
            "Options:\n" + "\n".join([f"{k}. {v}" for k, v in options.items()])
        )

        state = asyncio.run(self._run_graph(prompt))
        explanation = self._extract_explanation(state)
        selector_input_question = question + (f"\n\nContext from multi-agent reasoning:\n{explanation}" if explanation else "")
        selection = self.selector.predict(selector_input_question, options)

        plan_conf = self._extract_plan_confidence(state)
        probs = selection["probabilities"]
        confidence = probs.get(selection["choice"], 0.0)
        if plan_conf is not None:
            confidence *= max(0.0, min(1.0, plan_conf))

        return {
            "choice": selection["choice"],
            "probabilities": probs,
            "explanation": explanation or selection.get("explanation", ""),
            "confidence_override": confidence,
        }


class ExplanationJudge:
    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 3):
        self.llm = ChatOllama(**_ollama_kwargs(model, temperature))
        self.max_retries = max_retries

    def score(self, question: str, options: Dict[str, str], gold: str, predicted: str, explanation: str) -> Dict:
        option_block = "\n".join([f"{k}. {v}" for k, v in options.items()])
        prompt = (
            f"{PROMPT_JUDGE}\n\nQuestion:\n{question}\n\nOptions:\n{option_block}\n"
            f"Gold answer: {gold}\nModel choice: {predicted}\nExplanation: {explanation}\n"
            "Respond with JSON only."
        )
        last_error: Optional[str] = None
        for _ in range(self.max_retries):
            msg = self.llm.invoke(prompt)
            content = getattr(msg, "content", "") or ""
            parsed = safe_json_parse(content)
            if parsed and self._valid(parsed):
                return parsed
            last_error = content
        # If the judge refuses or keeps failing, return a neutral/low score instead of aborting the run.
        return {"plausibility": 0.0, "consistency": 0.0, "comment": f"unparsable: {last_error}"}

    @staticmethod
    def _valid(payload: Dict) -> bool:
        return (
            isinstance(payload.get("plausibility"), (int, float))
            and isinstance(payload.get("consistency"), (int, float))
        )


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


def expected_calibration_error(records: Iterable[Dict], bins: int = 10) -> float:
    buckets = [0 for _ in range(bins)]
    conf_sum = [0.0 for _ in range(bins)]
    correct_sum = [0.0 for _ in range(bins)]
    for r in records:
        conf = r.get("confidence", 0.0)
        correct = 1.0 if r["pred"] == r["gold"] else 0.0
        idx = min(bins - 1, int(conf * bins))
        buckets[idx] += 1
        conf_sum[idx] += conf
        correct_sum[idx] += correct
    ece = 0.0
    total = sum(buckets)
    for i in range(bins):
        if buckets[i] == 0:
            continue
        avg_conf = conf_sum[i] / buckets[i]
        avg_acc = correct_sum[i] / buckets[i]
        ece += (buckets[i] / total) * abs(avg_conf - avg_acc)
    return ece


def reliability_score(records: Iterable[Dict]) -> float:
    scores: List[float] = []
    for r in records:
        judge = r.get("judge", {})
        plaus = float(judge.get("plausibility", 0.0)) if judge else 0.0
        cons = float(judge.get("consistency", 0.0)) if judge else 0.0
        combined = (plaus + cons) / 2.0
        scores.append(0.5 * r.get("confidence", 0.0) + 0.5 * combined)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(
    dataset_root: Path,
    region: str,
    split: str,
    model: str,
    judge_model: str,
    limit: Optional[int],
    output: Optional[Path],
    runner_name: str,
) -> None:
    examples = load_examples(dataset_root, region, split, limit)
    total = len(examples)
    print(f"Starting eval: runner={runner_name}, model={model}, judge={judge_model}, split={region}/{split}, n={total}")
    runner = MultiAgentRunner(model) if runner_name == "multiagent" else OllamaMCQRunner(model)
    judge = ExplanationJudge(judge_model)
    records: List[Dict] = []
    for idx, ex in enumerate(examples, start=1):
        if idx == 1:
            print(f"Processing {idx}/{total}")
        else:
            print(f"Processing {idx}/{total}", end="\r", flush=True)
        question = ex["question"]
        options = normalize_options(ex["options"])
        gold = ex.get("answer_idx") or ex.get("answer")
        result = runner.predict(question, options)
        probs = result["probabilities"]
        confidence = result.get("confidence_override")
        if confidence is None:
            confidence = probs.get(result["choice"], 0.0)
        judge_scores = judge.score(question, options, gold, result["choice"], result.get("explanation", ""))
        records.append(
            {
                "question": question,
                "options": options,
                "gold": gold,
                "pred": result["choice"],
                "probabilities": probs,
                "confidence": confidence,
                "explanation": result.get("explanation", ""),
                "judge": judge_scores,
            }
        )

    labels = sorted({k for r in records for k in r["options"].keys()})
    acc = accuracy(records)
    f1 = macro_f1(records)
    brier = brier_score(records, labels)
    ece = expected_calibration_error(records)
    rel = reliability_score(records)

    summary = {
        "region": region,
        "split": split,
        "n": len(records),
        "accuracy": acc,
        "macro_f1": f1,
        "brier": brier,
        "ece": ece,
        "reliability": rel,
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
    parser = argparse.ArgumentParser(description="MedQA evaluation with Ollama baseline")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/data_clean/questions"))
    parser.add_argument("--region", type=str, default="US", choices=["US", "Taiwan", "Mainland"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--model", type=str, default=None, help="Ollama model name; defaults to $OLLAMA_MODEL or 'llama3.1'")
    parser.add_argument("--judge-model", type=str, default=None, help="LLM judge model; defaults to --model")
    parser.add_argument("--runner", type=str, default="baseline", choices=["baseline", "multiagent"], help="baseline=single LLM; multiagent=clinical pipeline + selector")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples (0 for full set)")
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSONL predictions")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    model = args.model or os.environ.get("OLLAMA_MODEL") or "llama3.1"
    judge_model = args.judge_model or model
    limit = args.limit if args.limit and args.limit > 0 else None
    output = args.output or Path(f"scripts/outputs/medqa_{args.region}_{args.split}.jsonl")
    evaluate(
        dataset_root=args.dataset_root,
        region=args.region,
        split=args.split,
        model=model,
        judge_model=judge_model,
        limit=limit,
        output=output,
        runner_name=args.runner,
    )


if __name__ == "__main__":
    main()
