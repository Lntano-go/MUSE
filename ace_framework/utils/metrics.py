"""Metrics tracking and management"""
import csv
import json
from typing import Dict, Any, List
from pathlib import Path


class MetricsTracker:
    """Tracks performance metrics during ACE execution"""
    
    def __init__(self):
        self.processed = 0
        self.correct = 0
        self.f1_total = 0.0
        self.f1_count = 0
        self.playbook_updates = 0
        self.history: List[Dict[str, Any]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.generator_prompt_tokens = 0
        self.generator_completion_tokens = 0
        self.reflection_prompt_tokens = 0
        self.reflection_completion_tokens = 0
        self.finish_reason_counts: Dict[str, int] = {}
    
    def record_result(
        self,
        question: str,
        predicted: str,
        ground_truth: str,
        is_correct: bool,
        playbook_size: int,
        used_skill_ids: List[str] = None,
        sample_id: str = "",
        context: str = "",
        reasoning: str = "",
        finish_reason: str = "",
        parse_failed: bool = False,
        truncated: bool = False,
        usage: Dict[str, int] = None,
        retrieved_skill_ids: List[str] = None,
        activated_skill_ids: List[str] = None,
        reflection_usage: Dict[str, int] = None,
        source_index: int = None,
        raw_completion: str = "",
    ) -> None:
        """Record a single result"""
        self.processed += 1
        if is_correct:
            self.correct += 1
        usage = usage or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        reflection_usage = reflection_usage or {}
        reflection_prompt_tokens = int(reflection_usage.get("prompt_tokens", 0) or 0)
        reflection_completion_tokens = int(reflection_usage.get("completion_tokens", 0) or 0)
        total_prompt_tokens = prompt_tokens + reflection_prompt_tokens
        total_completion_tokens = completion_tokens + reflection_completion_tokens
        self.total_prompt_tokens += total_prompt_tokens
        self.total_completion_tokens += total_completion_tokens
        self.generator_prompt_tokens += prompt_tokens
        self.generator_completion_tokens += completion_tokens
        self.reflection_prompt_tokens += reflection_prompt_tokens
        self.reflection_completion_tokens += reflection_completion_tokens
        finish_reason = finish_reason or "unknown"
        self.finish_reason_counts[finish_reason] = self.finish_reason_counts.get(finish_reason, 0) + 1
        
        self.history.append({
            "step": self.processed,
            "source_index": source_index,
            "sample_id": sample_id,
            "question": question[:100],  # Truncate for space
            "context": context,
            "reasoning": reasoning,
            "raw_completion": raw_completion,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "parse_failed": bool(parse_failed),
            "truncated": bool(truncated),
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reflection_prompt_tokens": reflection_prompt_tokens,
            "reflection_completion_tokens": reflection_completion_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "accuracy": self.get_accuracy(),
            "playbook_size": playbook_size,
            "retrieved_skill_ids": retrieved_skill_ids or [],
            "activated_skill_ids": activated_skill_ids or used_skill_ids or [],
            "used_skill_ids": used_skill_ids or []
        })
    
    def record_playbook_update(self) -> None:
        """Record a playbook update"""
        self.playbook_updates += 1

    def record_f1(self, f1: float) -> None:
        """Accumulate a per-sample F1 score when the dataset defines one."""
        if f1 is None:
            return
        self.f1_total += float(f1)
        self.f1_count += 1

    def record_auxiliary_usage(self, usage: Dict[str, int]) -> None:
        """Attach post-prediction reflection/curation usage to the latest task."""
        if not self.history:
            return
        usage = usage or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        record = self.history[-1]
        record["reflection_prompt_tokens"] += prompt_tokens
        record["reflection_completion_tokens"] += completion_tokens
        record["total_prompt_tokens"] += prompt_tokens
        record["total_completion_tokens"] += completion_tokens
        record["total_tokens"] += prompt_tokens + completion_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.reflection_prompt_tokens += prompt_tokens
        self.reflection_completion_tokens += completion_tokens
    
    def get_accuracy(self) -> float:
        """Calculate current accuracy"""
        if self.processed == 0:
            return 0.0
        return (self.correct / self.processed) * 100
    
    def get_accuracy_history(self) -> List[float]:
        """Get accuracy at each step"""
        return [record["accuracy"] for record in self.history]
    
    def get_playbook_size_history(self) -> List[int]:
        """Get playbook size at each step"""
        return [record["playbook_size"] for record in self.history]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_processed": self.processed,
            "total_correct": self.correct,
            "final_accuracy": self.get_accuracy(),
            "exact_match": self.get_accuracy(),
            "mean_f1": self.f1_total / self.f1_count if self.f1_count else None,
            "f1_scored_samples": self.f1_count,
            "playbook_updates": self.playbook_updates,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "generator_prompt_tokens": self.generator_prompt_tokens,
            "generator_completion_tokens": self.generator_completion_tokens,
            "reflection_prompt_tokens": self.reflection_prompt_tokens,
            "reflection_completion_tokens": self.reflection_completion_tokens,
            "mean_prompt_tokens_per_task": self.total_prompt_tokens / self.processed if self.processed else 0.0,
            "mean_completion_tokens_per_task": self.total_completion_tokens / self.processed if self.processed else 0.0,
            "parse_failures": sum(bool(row.get("parse_failed")) for row in self.history),
            "truncated_outputs": sum(bool(row.get("truncated")) for row in self.history),
            "finish_reason_counts": self.finish_reason_counts,
            "accuracy_history": self.get_accuracy_history(),
            "playbook_size_history": self.get_playbook_size_history()
        }
    
    def save(self, filepath: str) -> None:
        """Save metrics to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump({
                "summary": self.get_summary(),
                "detailed_history": self.history
            }, f, indent=2)

    def save_table(self, filepath: str) -> None:
        """Save detailed results as a CSV table"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "step",
            "source_index",
            "question",
            "sample_id",
            "predicted",
            "ground_truth",
            "correct",
            "exact_match",
            "f1",
            "parse_failed",
            "truncated",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "reflection_prompt_tokens",
            "reflection_completion_tokens",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "raw_completion",
            "accuracy",
            "playbook_size",
            "retrieved_skill_ids",
            "activated_skill_ids",
            "used_skill_ids"
        ]
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.history:
                writer.writerow({field: record.get(field, "") for field in fieldnames})
    
    def load(self, filepath: str) -> None:
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        summary = data["summary"]
        self.processed = summary["total_processed"]
        self.correct = summary["total_correct"]
        self.f1_count = int(summary.get("f1_scored_samples", 0))
        self.f1_total = float(summary.get("mean_f1", 0.0) or 0.0) * self.f1_count
        self.playbook_updates = summary["playbook_updates"]
        self.total_prompt_tokens = int(summary.get("prompt_tokens", 0))
        self.total_completion_tokens = int(summary.get("completion_tokens", 0))
        self.generator_prompt_tokens = int(summary.get("generator_prompt_tokens", 0))
        self.generator_completion_tokens = int(summary.get("generator_completion_tokens", 0))
        self.reflection_prompt_tokens = int(summary.get("reflection_prompt_tokens", 0))
        self.reflection_completion_tokens = int(summary.get("reflection_completion_tokens", 0))
        self.finish_reason_counts = summary.get("finish_reason_counts", {})
        self.history = data["detailed_history"]
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"Metrics(processed={self.processed}, "
            f"correct={self.correct}, "
            f"accuracy={self.get_accuracy():.1f}%, "
            f"updates={self.playbook_updates})"
        )
