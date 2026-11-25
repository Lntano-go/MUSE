"""Skill model - represents a high-level reusable reasoning skill"""
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Skill:
    """Represents a reusable skill distilled from the playbook"""
    id: str
    name: str
    description: str
    trigger_conditions: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    verification_checks: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    source_bullets: List[str] = field(default_factory=list)
    confidence: float = 0.5
    usage_count: int = 0
    helpful: int = 0
    harmful: int = 0
    retrieval_score: float = 0.0
    activated_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "trigger_keywords": self.trigger_keywords,
            "steps": self.steps,
            "verification_checks": self.verification_checks,
            "anti_patterns": self.anti_patterns,
            "source_bullets": self.source_bullets,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "retrieval_score": self.retrieval_score,
            "activated_count": self.activated_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create skill from dictionary"""
        return cls(
            id=data["id"],
            name=data.get("name", "Unnamed Skill"),
            description=data.get("description", ""),
            trigger_conditions=data.get("trigger_conditions", []),
            trigger_keywords=data.get("trigger_keywords", []),
            steps=data.get("steps", []),
            verification_checks=data.get("verification_checks", []),
            anti_patterns=data.get("anti_patterns", []),
            source_bullets=data.get("source_bullets", []),
            confidence=float(data.get("confidence", 0.5)),
            usage_count=int(data.get("usage_count", 0)),
            helpful=int(data.get("helpful", 0)),
            harmful=int(data.get("harmful", 0)),
            retrieval_score=float(data.get("retrieval_score", 0.0)),
            activated_count=int(data.get("activated_count", 0)),
        )

    def update_feedback(self, is_helpful: bool) -> None:
        """Update skill feedback counts."""
        if is_helpful:
            self.helpful += 1
        else:
            self.harmful += 1

    def get_score(self) -> float:
        """Calculate observed skill usefulness."""
        total = self.helpful + self.harmful
        if total == 0:
            return 0.0
        return (self.helpful - self.harmful) / total

    def __str__(self) -> str:
        """String representation for prompts"""
        triggers = "; ".join(self.trigger_conditions) if self.trigger_conditions else "General use"
        keywords = ", ".join(self.trigger_keywords) if self.trigger_keywords else "None"
        steps = "\n".join([f"  {i + 1}. {step}" for i, step in enumerate(self.steps)])
        checks = "\n".join([f"  - {check}" for check in self.verification_checks]) or "  - Verify the final answer is directly supported by the context."
        avoid = "\n".join([f"  - {pattern}" for pattern in self.anti_patterns]) or "  - Do not guess beyond the context."
        return (
            f"[{self.id}] {self.name} (confidence={self.confidence:.2f}, used={self.usage_count}, score={self.get_score():.2f})\n"
            f"Description: {self.description}\n"
            f"Trigger: {triggers}\n"
            f"Trigger keywords: {keywords}\n"
            f"Steps:\n{steps}\n"
            f"Verification checks:\n{checks}\n"
            f"Avoid:\n{avoid}"
        )
