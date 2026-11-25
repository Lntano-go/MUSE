"""Generator agent - generates answers using playbook"""
import re
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from ..models.playbook import Playbook
from ..models.skill_library import SkillLibrary


class Generator(BaseAgent):
    """Generates answers using current playbook"""

    def __init__(self, config):
        super().__init__(config)
        self.last_retrieved_skill_ids = []
        self.last_activated_skill_ids = []

    def _build_prompt(
        self,
        question: str,
        context: str,
        playbook: Playbook,
        skill_library: Optional[SkillLibrary] = None,
        line: str = "baseline",
        reflexion_memory: str = "",
        skill_top_k: int = 3,
        dataset_name: str = "",
        retrieval_pool_size: int = 20,
        use_uncertainty_gate: bool = True,
        gate_threshold: float = 0.45,
        gate_uncertainty_adjustment: float = 0.10,
        gate_helpfulness_adjustment: float = 0.05,
        gate_alpha: float = 0.45,
        gate_beta: float = 0.25,
        gate_gamma: float = 0.20,
        gate_delta: float = 0.10,
        record_skill_statistics: bool = True,
    ) -> str:
        """Build generation prompt"""
        self.last_retrieved_skill_ids = []
        self.last_activated_skill_ids = []
        if line == "base_llm":
            return self._build_answer_prompt(
                question, context, dataset_name, self._get_dataset_prompt_skill(dataset_name, ""),
                "Reason carefully using the supplied problem and context. Show the reasoning needed to support the answer.",
            )

        if line == "cot":
            return self._build_answer_prompt(
                question, context, dataset_name, self._get_dataset_prompt_skill(dataset_name, ""),
                "Think step by step, break down sub-questions if needed, and support the final answer with the supplied input.",
            )

        if line == "reflexion":
            memory_text = reflexion_memory if reflexion_memory else "No reflection memory yet."
            return self._build_answer_prompt(
                question, context, dataset_name, self._get_dataset_prompt_skill(dataset_name, ""),
                "Use reflection memory when relevant, reason step by step, and support the answer with the supplied input.",
                additional_guidance=f"Reflection Memory from previous failed attempts:\n{memory_text}",
            )

        use_macro_skillline = skill_library is not None and line == "skillline"
        playbook_text = "" if use_macro_skillline else playbook.get_formatted_playbook()
        skills_text = ""
        skill_instruction = ""
        skill_schema = ""
        if use_macro_skillline:
            task_type = self._infer_task_type(question)
            retrieval_query = f"Task type: {task_type}\nQuestion: {question}\nContext: {context[:2500]}"
            candidates = skill_library.retrieve_candidates(
                retrieval_query, top_k=retrieval_pool_size,
                alpha=gate_alpha, beta=gate_beta, gamma=gate_gamma, delta=gate_delta,
                update_statistics=record_skill_statistics,
            )
            activated = skill_library.activate_candidates(
                candidates, skill_top_k, use_uncertainty_gate, gate_threshold,
                gate_uncertainty_adjustment, gate_helpfulness_adjustment,
                record_activation=record_skill_statistics,
            )
            self.last_retrieved_skill_ids = [skill.id for skill in candidates]
            self.last_activated_skill_ids = [skill.id for skill in activated]
            skills_text = "\n\n" + ("\n\n".join(str(skill) for skill in activated) if activated else "No skills passed the activation gate.")
            skill_instruction = f"""
7. MACRO SKILLLINE MODE: Use skills as optional high-level guidance, not as mandatory templates.
    a. The detected task type is: {task_type}.
    b. Apply retrieved macro skills only when they are clearly relevant to this question.
    c. Do not force a skill. If direct reasoning is safer, solve directly and leave used_skill_ids empty.
    d. If a skill is used, briefly state how it shaped the equation setup or verification.
    e. Always follow the Dataset Prompt Skill because it matches the current dataset format.
    f. For math problems, finish with a quick recomputation/substitution check and return only the simplified final answer.
"""
            skill_schema = ',\n    "used_skill_ids": ["skill-00001"]'
        elif skill_library is not None:
            task_type = self._infer_task_type(question)
            retrieval_query = f"Task type: {task_type}\nQuestion: {question}\nContext: {context[:2500]}"
            candidates = skill_library.retrieve_candidates(
                retrieval_query, top_k=retrieval_pool_size,
                alpha=gate_alpha, beta=gate_beta, gamma=gate_gamma, delta=gate_delta,
                update_statistics=record_skill_statistics,
            )
            activated = candidates[:skill_top_k]
            self.last_retrieved_skill_ids = [skill.id for skill in candidates]
            self.last_activated_skill_ids = [skill.id for skill in activated]
            skills_text = "\n\n" + "\n\n".join(str(skill) for skill in activated)
            skill_instruction = f"\n7. If generated skills are available, use only clearly relevant skills. Do not force skill use."
            skill_schema = ',\n    "used_skill_ids": ["skill-00001"]'
        
        answer_contract = self._get_dataset_prompt_skill(dataset_name, "")

        guidance = "Reason step by step and apply relevant playbook strategies when present."
        return self._build_answer_prompt(
            question,
            context,
            dataset_name,
            "\n\n".join(part for part in (answer_contract, playbook_text, skills_text) if part),
            guidance,
            additional_guidance=skill_instruction,
            answer_schema=f'"used_bullet_ids": ["ctx-00001"],\n    "used_skill_ids": ["skill-00001"],\n    ' if skill_schema else '"used_bullet_ids": [],\n    "used_skill_ids": [],\n    ',
        )
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse generation response"""
        return {
            "reasoning": response.get("reasoning", ""),
            "used_bullet_ids": response.get("used_bullet_ids", []),
            "used_skill_ids": response.get("used_skill_ids", []),
            "final_answer": str(response.get("final_answer", "")),
            "_usage": response.get("_usage", {}),
            "_finish_reason": response.get("_finish_reason"),
            "_response_chars": response.get("_response_chars", 0),
            "_raw_content": response.get("_raw_content", ""),
        }
    
    def _get_error_response(self) -> Dict[str, Any]:
        """Get error response for generation"""
        return {
            "reasoning": "Error in generation",
            "used_bullet_ids": [],
            "used_skill_ids": [],
            "final_answer": "",
            "_usage": dict(self.llm_client.last_usage),
            "_finish_reason": "request_failed_or_unparseable",
            "_response_chars": 0,
            "_raw_content": "",
        }

    @staticmethod
    def _infer_task_type(question: str) -> str:
        """Infer a coarse math/reasoning task type to improve skill retrieval."""
        text = question.lower()
        patterns = [
            (r"base\s*\d|_\d|binary|octal|hexadecimal", "base arithmetic / base conversion"),
            (r"slope|coordinate|midpoint|distance|line passing|triangle|circle|angle|geometry", "coordinate or geometric reasoning"),
            (r"probability|chance|random|ways|arrange|permutation|combination|choose", "probability or counting"),
            (r"divisor|factor|prime|gcd|lcm|multiple|modulo|remainder", "number theory / divisibility"),
            (r"exponent|power|root|sqrt|log|radical", "exponents, roots, or logarithms"),
            (r"solve|equation|inequality|function|domain|piecewise", "algebraic equation or inequality"),
            (r"average|mean|median|ratio|percent|fraction", "arithmetic word problem"),
        ]
        for pattern, task_type in patterns:
            if re.search(pattern, text):
                return task_type
        return "general mathematical reasoning"

    @staticmethod
    def _get_role_text(dataset_name: str, use_macro_skillline: bool) -> str:
        """Return a dataset-matched system role line."""
        normalized = (dataset_name or "").lower().replace("_", "-")
        if normalized in {"gsm-hard", "gsmhard", "gsm8k"}:
            return "You are an expert at solving arithmetic word problems with exact numerical reasoning."
        if normalized == "math":
            return "You are an expert at solving competition-style math problems."
        if normalized in {"hotpot-dev-distractor-v1", "hotpotqa"}:
            return "You are an expert at evidence-grounded multi-hop question answering."
        if use_macro_skillline:
            return "You are an expert reasoner using optional macro skills as high-level guidance."
        return "You are an expert at answering multi-hop reasoning questions."

    @staticmethod
    def _get_dataset_prompt_skill(dataset_name: str, task_type: str) -> str:
        """Return dataset-matched guidance as a built-in prompt skill."""
        normalized = (dataset_name or "").lower().replace("_", "-")
        if normalized in {"gsm-hard", "gsmhard"}:
            return f"""
## Dataset Prompt Skill [dataset-gsm-hard]
Name: GSM-Hard exact-number word-problem protocol
Trigger: GSM-Hard arithmetic word problems with unusually large, negative, fractional, or unrealistic values.
Task type: {task_type}
Guidance:
  1. Use every number exactly as written in the question; never replace it with a familiar GSM8K template value.
  2. Translate the story into arithmetic equations before calculating.
  3. Large, negative, zero, or fractional answers may be valid; do not reject them for being unrealistic.
  4. Preserve units and operation order from the text.
  5. Return only the final numeric answer.
Verification:
  - Check that each numeric value used appears in the question.
  - Recompute the final arithmetic once before answering.
"""
        if normalized == "math":
            return f"""
## Dataset Prompt Skill [dataset-math]
Name: MATH concise solution protocol
Trigger: Competition-style math problems.
Task type: {task_type}
Guidance:
  1. Identify the relevant formula, theorem, or algebraic structure first.
  2. Keep symbolic transformations consistent with domain constraints.
  3. Verify boundary cases, signs, and simplification before finalizing.
  4. Return only the final simplified answer.
"""
        if normalized in {"gsm8k"}:
            return f"""
## Dataset Prompt Skill [dataset-gsm8k]
Name: GSM8K step-by-step arithmetic protocol
Trigger: Grade-school arithmetic word problems.
Task type: {task_type}
Guidance:
  1. Define quantities from the story.
  2. Compute intermediate values in the same order as the events.
  3. Verify the final answer has the requested unit.
  4. Return only the final numeric answer.
"""
        if normalized in {"hotpot-dev-distractor-v1", "hotpotqa"}:
            return f"""
## Dataset Prompt Skill [dataset-hotpotqa]
Name: HotpotQA evidence-grounded multi-hop protocol
Trigger: Multi-hop question answering with distractor context.
Task type: {task_type}
Guidance:
  1. Identify the bridge entity and the target entity.
  2. Use only facts supported by the context.
  3. Avoid distractor entities with similar names.
  4. Return the short answer only.
"""
        return ""
    
    def generate(
        self,
        question: str,
        context: str,
        playbook: Playbook,
        skill_library: Optional[SkillLibrary] = None,
        line: str = "baseline",
        reflexion_memory: str = "",
        skill_top_k: int = 3,
        dataset_name: str = "",
        retrieval_pool_size: int = 20,
        use_uncertainty_gate: bool = True,
        gate_threshold: float = 0.45,
        gate_uncertainty_adjustment: float = 0.10,
        gate_helpfulness_adjustment: float = 0.05,
        gate_alpha: float = 0.45,
        gate_beta: float = 0.25,
        gate_gamma: float = 0.20,
        gate_delta: float = 0.10,
        record_skill_statistics: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate answer with reasoning
        
        Args:
            question: Question to answer
            context: Supporting context
            playbook: Current playbook
            skill_library: Optional generated skill library
            line: Experiment line
            reflexion_memory: Optional reflection memory for Reflexion baseline
            skill_top_k: Number of retrieved skills injected into prompt
            
        Returns:
            Dictionary with reasoning, used_bullet_ids, and final_answer
        """
        return self.execute(
            question=question,
            context=context,
            playbook=playbook,
            skill_library=skill_library,
            line=line,
            reflexion_memory=reflexion_memory,
            skill_top_k=skill_top_k,
            dataset_name=dataset_name,
            retrieval_pool_size=retrieval_pool_size,
            use_uncertainty_gate=use_uncertainty_gate,
            gate_threshold=gate_threshold,
            gate_uncertainty_adjustment=gate_uncertainty_adjustment,
            gate_helpfulness_adjustment=gate_helpfulness_adjustment,
            gate_alpha=gate_alpha,
            gate_beta=gate_beta,
            gate_gamma=gate_gamma,
            gate_delta=gate_delta,
            record_skill_statistics=record_skill_statistics,
        )

    def _build_answer_prompt(
        self,
        question: str,
        context: str,
        dataset_name: str,
        answer_contract: str,
        reasoning_instruction: str,
        additional_guidance: str = "",
        answer_schema: str = '"used_bullet_ids": [],\n    "used_skill_ids": [],\n    ',
    ) -> str:
        """Keep task and final-answer instructions consistent across baselines."""
        return f"""{self._get_role_text(dataset_name, False)}

{answer_contract}

{additional_guidance}

Question: {question}

Supporting Context: {context}

Instructions:
1. {reasoning_instruction}
2. Put only the answer itself in final_answer, without surrounding explanation.
3. Return a concise answer in the dataset's standard answer form.

Respond in JSON format:
{{
    "reasoning": "Brief rationale",
    {answer_schema}
    "final_answer": "Short direct answer here"
}}
"""
