"""SkillGenerator agent - distills high-level reusable skills from playbook or trajectories"""
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..models.playbook import Playbook


class SkillGenerator(BaseAgent):
    """Generates reusable skills from accumulated playbook strategies"""

    def _build_prompt(
        self,
        playbook: Playbook = None,
        max_skills: int = 5,
        trajectories: List[Dict[str, Any]] = None,
        dataset_name: str = ""
    ) -> str:
        """Build skill generation prompt"""
        if trajectories is not None:
            return self._build_trajectory_prompt(trajectories, dataset_name, max_skills)

        return f"""You are a skill distillation expert. Convert the current playbook into a compact set of reusable reasoning skills.

Current Playbook:
{playbook.get_formatted_playbook()}

Instructions:
1. Distill only stable, reusable skills that can improve future answers beyond simply rereading the playbook.
2. Merge duplicate or overlapping bullets into one executable procedure.
3. Each skill must contain concrete trigger conditions, trigger keywords, step-by-step actions, verification checks, and anti-patterns to avoid.
4. Prefer skills that prevent repeated errors: entity confusion, wrong bridge entity, insufficient evidence, answer-format mismatch, arithmetic/calculation slips, and unsupported guessing.
5. Generate at most {max_skills} skills, ordered by expected usefulness.
6. Use source_bullets to record the playbook bullet IDs supporting the skill.
7. Make each step operational: tell the solver exactly what to inspect, compare, or verify in the context.

Respond in JSON format:
{{
    "skills": [
        {{
            "name": "Skill name",
            "description": "What this skill helps with",
            "trigger_conditions": ["When to use this skill"],
            "trigger_keywords": ["keyword or phrase that suggests using this skill"],
            "steps": ["Step 1", "Step 2", "Step 3"],
            "verification_checks": ["Check that must pass before final_answer"],
            "anti_patterns": ["Common mistake this skill prevents"],
            "source_bullets": ["ctx-00001", "ctx-00002"],
            "confidence": 0.8
        }}
    ]
}}
"""

    def _build_trajectory_prompt(
        self,
        trajectories: List[Dict[str, Any]],
        dataset_name: str,
        max_skills: int = 5
    ) -> str:
        """Build macro skill generation prompt from recent answer trajectories."""
        examples = []
        for index, item in enumerate(trajectories[-max_skills * 4:], start=1):
            sample = item.get("sample", {})
            generation = item.get("generation", {})
            feedback = item.get("feedback", {})
            outcome = "correct" if feedback.get("correct") is True else "incorrect" if feedback.get("correct") is False else "unknown"
            examples.append(
                f"Example {index}:\n"
                f"Question: {sample.get('question', '')}\n"
                f"Predicted: {generation.get('final_answer', '')}\n"
                f"Reasoning summary: {generation.get('reasoning', '')[:800]}\n"
                f"Observed outcome: {outcome}\n"
                f"Post-answer insight: {feedback.get('key_insight', '')[:500]}\n"
                f"Root cause: {feedback.get('root_cause', '')[:500]}\n"
            )

        return f"""You are a macro-skill distillation expert for dataset-specific reasoning.

Dataset: {dataset_name}

Recent answer trajectories:
{chr(10).join(examples)}

Goal:
Create high-level reusable guidance skills that improve future answers on this dataset.

Important constraints:
1. Generate macro skills, not problem-specific tips.
2. Do not memorize exact questions or answers.
3. Prefer dataset-level reasoning protocols and recurring error-prevention skills.
4. For GSM-Hard, emphasize exact-number extraction, symbolic equation setup, unrealistic value tolerance, negative/fractional answer tolerance, and arithmetic verification.
5. For MATH, emphasize formula selection, domain checks, sign checks, and final simplification.
6. For HotpotQA, emphasize bridge-entity selection, evidence grounding, and distractor avoidance.
7. These skills will be optional guidance; each skill must be useful without being mandatory.
8. Use observed correctness and post-answer insights to distinguish effective strategies from recurring errors. Do not copy exact questions, predictions, or answers into a skill.
9. Generate at most {max_skills} skills.

Respond in JSON format:
{{
    "skills": [
        {{
            "name": "Macro skill name",
            "description": "High-level reusable guidance",
            "trigger_conditions": ["When this guidance applies"],
            "trigger_keywords": ["dataset/task keywords"],
            "steps": ["Guidance step 1", "Guidance step 2", "Guidance step 3"],
            "verification_checks": ["Check before final answer"],
            "anti_patterns": ["Mistake this prevents"],
            "source_bullets": [],
            "confidence": 0.8
        }}
    ]
}}
"""

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse generated skills"""
        valid_skills = []
        for skill in self._iter_skill_dicts(response.get("skills", [])):
            if not skill.get("name") or not skill.get("description"):
                continue
            try:
                confidence = float(skill.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            valid_skills.append({
                "name": skill.get("name", "Unnamed Skill"),
                "description": skill.get("description", ""),
                "trigger_conditions": skill.get("trigger_conditions", []),
                "trigger_keywords": skill.get("trigger_keywords", []),
                "steps": skill.get("steps", []),
                "verification_checks": skill.get("verification_checks", []),
                "anti_patterns": skill.get("anti_patterns", []),
                "source_bullets": skill.get("source_bullets", []),
                "confidence": confidence
            })
        return {"skills": valid_skills}

    @classmethod
    def _iter_skill_dicts(cls, value):
        """Yield skill objects from occasionally nested repaired JSON arrays."""
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from cls._iter_skill_dicts(item)

    def _get_error_response(self) -> Dict[str, Any]:
        """Get default error response"""
        return {"skills": []}

    def generate_skills(self, playbook: Playbook, max_skills: int = 5) -> List[Dict[str, Any]]:
        """Generate reusable skills from the current playbook"""
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if len(playbook) == 0:
            return []
        result = self.execute(playbook=playbook, max_skills=max_skills)
        return result.get("skills", [])

    def generate_macro_skills(
        self,
        trajectories: List[Dict[str, Any]],
        dataset_name: str,
        max_skills: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate macro reusable skills from recent answer trajectories."""
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not trajectories:
            return []
        result = self.execute(
            trajectories=trajectories,
            dataset_name=dataset_name,
            max_skills=max_skills
        )
        self.last_usage = getattr(self, "last_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return result.get("skills", [])

