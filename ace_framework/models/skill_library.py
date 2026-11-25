"""Skill library - manages generated skills"""
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, List, Optional
from .skill import Skill


class SkillLibrary:
    """Collection of reusable skills distilled from the playbook"""

    def __init__(self):
        self.skills: List[Skill] = []
        self.next_id: int = 1

    def replace_skills(self, skill_dicts: List[Dict[str, Any]]) -> None:
        """Replace the active library while carrying feedback for stable names."""
        previous_usage = {
            self._canonical_name(skill.name): (skill.usage_count, skill.helpful, skill.harmful)
            for skill in self.skills
        }
        self.skills = []
        self.next_id = 1
        for skill_dict in skill_dicts:
            skill = self.add_skill(
                name=skill_dict.get("name", "Unnamed Skill"),
                description=skill_dict.get("description", ""),
                trigger_conditions=skill_dict.get("trigger_conditions", []),
                trigger_keywords=skill_dict.get("trigger_keywords", []),
                steps=skill_dict.get("steps", []),
                verification_checks=skill_dict.get("verification_checks", []),
                anti_patterns=skill_dict.get("anti_patterns", []),
                source_bullets=skill_dict.get("source_bullets", []),
                confidence=float(skill_dict.get("confidence", 0.5))
            )
            usage_count, helpful, harmful = previous_usage.get(self._canonical_name(skill.name), (0, 0, 0))
            skill.usage_count = usage_count
            skill.helpful = helpful
            skill.harmful = harmful

    def update_library(
        self,
        skill_dicts: List[Dict[str, Any]],
        max_library_size: int,
        retain_threshold: float,
        retire_threshold: float,
        utility_weights: tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        """Merge new candidates, then retain or retire skills by observed utility."""
        previous = {self._canonical_name(skill.name): skill for skill in self.skills}
        candidates: List[Skill] = []
        next_id = self.next_id

        for data in skill_dicts:
            name = data.get("name", "Unnamed Skill")
            prior = previous.get(self._canonical_name(name))
            skill = Skill(
                id=prior.id if prior else f"skill-{next_id:05d}",
                name=name,
                description=data.get("description", ""),
                trigger_conditions=data.get("trigger_conditions", []),
                trigger_keywords=data.get("trigger_keywords", []),
                steps=data.get("steps", []),
                verification_checks=data.get("verification_checks", []),
                anti_patterns=data.get("anti_patterns", []),
                source_bullets=data.get("source_bullets", []),
                confidence=float(data.get("confidence", 0.5)),
                usage_count=prior.usage_count if prior else 0,
                helpful=prior.helpful if prior else 0,
                harmful=prior.harmful if prior else 0,
            )
            candidates.append(skill)
            if prior is None:
                next_id += 1

        candidate_names = {self._canonical_name(skill.name) for skill in candidates}
        candidates.extend(
            skill for skill in self.skills
            if self._canonical_name(skill.name) not in candidate_names
        )

        confidence_weight, helpfulness_weight, usage_weight, redundancy_weight = utility_weights
        for skill in candidates:
            skill.retrieval_score = self._lexical_score(skill, skill.name + " " + skill.description)

        # Penalize duplicate guidance by lexical overlap in the skill descriptions.
        for skill in candidates:
            duplicate_count = sum(
                self._token_overlap(skill, other) >= 0.8
                for other in candidates if other is not skill
            )
            score = skill.get_score()
            normalized_usage = min(skill.usage_count, 5) / 5.0
            skill.retrieval_score = (
                confidence_weight * skill.confidence
                + helpfulness_weight * score
                + usage_weight * normalized_usage
                - redundancy_weight * duplicate_count
            )

        eligible = [skill for skill in candidates if skill.retrieval_score >= retire_threshold]
        ranked = sorted(eligible, key=lambda skill: skill.retrieval_score, reverse=True)
        retained = [skill for skill in ranked if skill.retrieval_score >= retain_threshold]
        retained_names = {self._canonical_name(skill.name) for skill in retained}
        if len(retained) < max_library_size:
            for skill in ranked:
                key = self._canonical_name(skill.name)
                if key not in retained_names:
                    retained.append(skill)
                    retained_names.add(key)
                if len(retained) >= max_library_size:
                    break
        retained = retained[:max_library_size]

        self.skills = retained
        self.next_id = max(
            [int(skill.id.split("-")[1]) for skill in retained] + [next_id - 1]
        ) + 1
        return {
            "candidate_count": len(candidates),
            "retained_count": len(retained),
            "retired_count": len(candidates) - len(retained),
            "retained_ids": [skill.id for skill in retained],
            "utility_scores": {skill.id: skill.retrieval_score for skill in candidates},
        }

    def add_skill(
        self,
        name: str,
        description: str,
        trigger_conditions: Optional[List[str]] = None,
        trigger_keywords: Optional[List[str]] = None,
        steps: Optional[List[str]] = None,
        verification_checks: Optional[List[str]] = None,
        anti_patterns: Optional[List[str]] = None,
        source_bullets: Optional[List[str]] = None,
        confidence: float = 0.5
    ) -> Skill:
        """Add a new skill"""
        skill = Skill(
            id=f"skill-{str(self.next_id).zfill(5)}",
            name=name,
            description=description,
            trigger_conditions=trigger_conditions or [],
            trigger_keywords=trigger_keywords or [],
            steps=steps or [],
            verification_checks=verification_checks or [],
            anti_patterns=anti_patterns or [],
            source_bullets=source_bullets or [],
            confidence=confidence
        )
        self.skills.append(skill)
        self.next_id += 1
        return skill

    def get_skill_by_id(self, skill_id: str) -> Optional[Skill]:
        """Retrieve skill by ID"""
        for skill in self.skills:
            if skill.id == skill_id:
                return skill
        return None

    def update_usage(self, skill_ids: List[str]) -> None:
        """Update usage counts for selected skills"""
        for skill_id in skill_ids:
            skill = self.get_skill_by_id(skill_id)
            if skill:
                skill.usage_count += 1

    def update_feedback(self, skill_id: str, is_helpful: bool) -> bool:
        """Update helpful/harmful counts for a skill."""
        skill = self.get_skill_by_id(skill_id)
        if skill:
            skill.update_feedback(is_helpful)
            return True
        return False

    def get_relevant_skills(self, query: str, top_k: int = 3) -> List[Skill]:
        """Rank skills by lexical relevance, confidence, and observed usefulness."""
        if not self.skills:
            return []

        return self.retrieve_candidates(query, top_k=top_k)

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 20,
        alpha: float = 0.45,
        beta: float = 0.25,
        gamma: float = 0.20,
        delta: float = 0.10,
        update_statistics: bool = True,
    ) -> List[Skill]:
        """Return a ranked candidate pool before the activation gate."""
        if not self.skills:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return sorted(self.skills, key=lambda skill: skill.confidence, reverse=True)[:top_k]

        scored_skills = []
        for skill in self.skills:
            trigger_tokens = self._tokenize(" ".join(skill.trigger_conditions + skill.trigger_keywords))
            description_tokens = self._tokenize(f"{skill.name} {skill.description}")
            procedure_tokens = self._tokenize(" ".join(skill.steps + skill.verification_checks + skill.anti_patterns))
            trigger_overlap = len(query_tokens.intersection(trigger_tokens))
            description_overlap = len(query_tokens.intersection(description_tokens))
            procedure_overlap = len(query_tokens.intersection(procedure_tokens))
            relevance = min(1.0, (trigger_overlap * 3.0 + description_overlap * 1.5 + procedure_overlap * 0.75) / 12.0)
            uncertainty = 1.0 - skill.confidence
            helpfulness = skill.get_score()
            score = alpha * relevance + beta * skill.confidence + gamma * helpfulness - delta * uncertainty
            if update_statistics:
                skill.retrieval_score = score
                scored_skill = skill
            else:
                scored_skill = replace(skill, retrieval_score=score)
            scored_skills.append((score, scored_skill))

        scored_skills.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored_skills[:top_k]]

    @staticmethod
    def activate_candidates(
        candidates: List[Skill],
        top_k: int,
        use_gate: bool,
        base_threshold: float,
        uncertainty_adjustment: float,
        helpfulness_adjustment: float,
        record_activation: bool = True,
    ) -> List[Skill]:
        """Apply a dynamic confidence/helpfulness threshold, or plain top-k ablation."""
        if not use_gate:
            activated = candidates[:top_k]
            if record_activation:
                for skill in activated:
                    skill.activated_count += 1
            return activated
        if not candidates:
            return []
        mean_uncertainty = sum(1.0 - skill.confidence for skill in candidates) / len(candidates)
        mean_helpfulness = sum(skill.get_score() for skill in candidates) / len(candidates)
        threshold = base_threshold + uncertainty_adjustment * mean_uncertainty - helpfulness_adjustment * mean_helpfulness
        activated = [skill for skill in candidates if skill.retrieval_score >= threshold][:top_k]
        if record_activation:
            for skill in activated:
                skill.activated_count += 1
        return activated

    @staticmethod
    def _skill_text(skill: Skill) -> str:
        return " ".join(
            [skill.name, skill.description]
            + skill.trigger_conditions + skill.trigger_keywords
            + skill.steps + skill.verification_checks + skill.anti_patterns
        )

    @classmethod
    def _token_overlap(cls, left: Skill, right: Skill) -> float:
        a = cls._tokenize(cls._skill_text(left))
        b = cls._tokenize(cls._skill_text(right))
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))

    @classmethod
    def _lexical_score(cls, skill: Skill, query: str) -> float:
        query_tokens = cls._tokenize(query)
        skill_tokens = cls._tokenize(cls._skill_text(skill))
        return len(query_tokens & skill_tokens) / max(1, len(query_tokens))

    def get_formatted_skills(self, query: str = "", top_k: Optional[int] = None) -> str:
        """Get formatted top-k relevant skills for prompts"""
        skills = self.skills
        if query and top_k is not None:
            skills = self.get_relevant_skills(query, top_k)

        if not skills:
            return "No skills generated yet."
        formatted = ["## Generated Skills"]
        for skill in skills:
            formatted.append(str(skill))
        return "\n\n".join(formatted)

    @staticmethod
    def _canonical_name(name: str) -> str:
        """Normalize skill names so usage survives regeneration."""
        return re.sub(r"\s+", " ", name.strip().lower())

    @staticmethod
    def _tokenize(text: str) -> set:
        """Tokenize text for simple lexical retrieval."""
        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
            "with", "by", "from", "is", "are", "was", "were", "what", "which",
            "who", "when", "where", "why", "how", "this", "that", "these", "those"
        }
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return {token for token in tokens if token not in stopwords and len(token) > 2}

    def get_statistics(self) -> Dict[str, Any]:
        """Get skill library statistics"""
        return {
            "total_skills": len(self.skills),
            "total_usage": sum(skill.usage_count for skill in self.skills),
            "total_helpful": sum(skill.helpful for skill in self.skills),
            "total_harmful": sum(skill.harmful for skill in self.skills),
            "avg_confidence": (
                sum(skill.confidence for skill in self.skills) / len(self.skills)
                if self.skills else 0.0
            )
        }

    def save(self, filepath: str) -> None:
        """Save skills to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([skill.to_dict() for skill in self.skills], f, indent=2)

    def load(self, filepath: str) -> None:
        """Load skills from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.skills = [Skill.from_dict(skill_data) for skill_data in data]
        if self.skills:
            max_id = max(int(skill.id.split("-")[1]) for skill in self.skills)
            self.next_id = max_id + 1
        else:
            self.next_id = 1

    def __len__(self) -> int:
        """Get number of skills"""
        return len(self.skills)
