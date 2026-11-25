"""Reflector agent - reflects on trajectories to extract insights"""
from typing import Dict, Any, List, Optional
from collections import Counter
import re
import string
from .base_agent import BaseAgent
from ..models.playbook import Playbook


class Reflector(BaseAgent):
    """Reflects on trajectories to extract insights"""
    
    def _build_prompt(
        self,
        question: str,
        reasoning: str,
        predicted: str,
        ground_truth: str,
        used_bullets: List[str],
        used_skills: List[str],
        playbook: Playbook,
        is_correct: bool,
        metric_name: str = "",
    ) -> str:
        """Build reflection prompt"""
        bullet_info = ""
        for bullet_id in used_bullets:
            bullet = playbook.get_bullet_by_id(bullet_id)
            if bullet:
                bullet_info += f"\n- [{bullet_id}]: {bullet.content}"
        
        return f"""You are an expert analyzer. Analyze this question-answering attempt.

Question: {question}
Model's Reasoning: {reasoning}
Predicted Answer: {predicted}
Ground Truth: {ground_truth}
Scoring metric: {metric_name}
Result: {"✓ CORRECT" if is_correct else "✗ INCORRECT"}

Bullets Used: {bullet_info if bullet_info else "None"}
Skills Used: {", ".join(used_skills) if used_skills else "None"}

Instructions:
Analyze the attempt and extract insights:
1. What strategies worked well or failed?
2. What new insights should be added to the playbook?
3. Tag each used bullet as helpful, harmful, or neutral
4. If skills were used, tag each used skill as helpful, harmful, or neutral based on whether it improved the answer.

Respond in JSON format:
{{
    "error_identification": "What went wrong or right",
    "root_cause": "Why this happened",
    "key_insight": "Actionable strategy to remember",
    "bullet_tags": [{{"id": "ctx-00001", "tag": "helpful"}}],
    "skill_tags": [{{"id": "skill-00001", "tag": "helpful"}}]
}}
"""
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse reflection response"""
        return {
            "error_identification": response.get("error_identification", ""),
            "root_cause": response.get("root_cause", ""),
            "key_insight": response.get("key_insight", ""),
            "bullet_tags": response.get("bullet_tags", []),
            "skill_tags": response.get("skill_tags", [])
        }
    
    def _get_error_response(self) -> Dict[str, Any]:
        """Get error response for reflection"""
        return {
            "error_identification": "Analysis error",
            "root_cause": "Unknown",
            "key_insight": "Unable to extract insight",
            "bullet_tags": [],
            "skill_tags": []
        }
    
    @staticmethod
    def check_answer(predicted: str, ground_truth: str) -> bool:
        """Conservative normalized exact match, not substring or token-overlap matching."""
        normalized_pred = Reflector._normalize_answer(predicted)
        return bool(normalized_pred and normalized_pred == Reflector._normalize_answer(ground_truth))

    @staticmethod
    def score_answer(predicted: str, ground_truth: str, metric_name: str) -> bool:
        """Apply a dataset-specific exact-match scorer with conservative fallbacks."""
        return Reflector.score_answer_metrics(predicted, ground_truth, metric_name)["exact_match"]

    @staticmethod
    def score_answer_metrics(predicted: str, ground_truth: str, metric_name: str) -> Dict[str, Any]:
        """Return exact match and, for HotpotQA, normalized token-level F1 in [0, 100]."""
        metric = (metric_name or "").lower().replace("_", "-")
        pred = Reflector._normalize_answer(predicted)
        gold = Reflector._normalize_answer(ground_truth)
        if metric in {"hotpotqa", "hotpot-dev-distractor-v1"}:
            normalized_pred = Reflector._normalize_hotpot(pred)
            normalized_gold = Reflector._normalize_hotpot(gold)
            exact_match = normalized_pred == normalized_gold
            predicted_tokens = normalized_pred.split()
            gold_tokens = normalized_gold.split()
            if not predicted_tokens or not gold_tokens:
                token_f1 = 100.0 * float(predicted_tokens == gold_tokens)
            else:
                common = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
                precision = common / len(predicted_tokens)
                recall = common / len(gold_tokens)
                token_f1 = (
                    100 * 2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
            return {"exact_match": exact_match, "f1": token_f1}
        if metric in {"gsm-hard", "gsmhard", "gsm8k"}:
            pred_number = Reflector._extract_number(pred)
            gold_number = Reflector._extract_number(gold)
            if pred_number is not None and gold_number is not None:
                exact_match = pred_number == gold_number
                return {"exact_match": exact_match, "f1": None}
        return {"exact_match": bool(pred and pred == gold), "f1": None}

    @staticmethod
    def _normalize_hotpot(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _extract_number(text: str):
        from decimal import Decimal, InvalidOperation
        text = Reflector._normalize_answer(text)
        text = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", text, flags=re.IGNORECASE)
        text = text.replace(",", "").replace("$", "").strip()
        if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    @staticmethod
    def _normalize_answer(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text)
        boxed = re.findall(r"\\boxed\s*\{([^{}]*)\}", text)
        if boxed:
            text = boxed[-1]
        if "####" in text:
            text = text.rsplit("####", 1)[-1]
        if "final answer:" in text:
            text = text.rsplit("final answer:", 1)[-1]
        text = text.replace(",", "")
        text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"[\s$]+$", "", text).strip()
    
    def reflect(
        self,
        question: str,
        reasoning: str,
        predicted: str,
        ground_truth: str,
        used_bullets: List[str],
        playbook: Playbook,
        used_skills: List[str] = None,
        dataset_name: str = "",
        metric_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze what went right or wrong
        
        Args:
            question: Original question
            reasoning: Model's reasoning
            predicted: Predicted answer
            ground_truth: Correct answer
            used_bullets: List of bullet IDs used
            playbook: Current playbook
            
        Returns:
            Dictionary with reflection analysis and correctness
        """
        metric = dataset_name if not metric_name or metric_name == "dataset_default" else metric_name
        if metric in {"gsm-hard", "math", "gsm8k"}:
            predicted = self._normalize_answer(predicted)
            ground_truth = self._normalize_answer(ground_truth)
        is_correct = self.score_answer(predicted, ground_truth, metric)
        
        result = self.execute(
            question=question,
            reasoning=reasoning,
            predicted=predicted,
            ground_truth=ground_truth,
            used_bullets=used_bullets,
            used_skills=used_skills or [],
            playbook=playbook,
            is_correct=is_correct,
            metric_name=metric,
        )
        
        result["is_correct"] = is_correct
        return result
