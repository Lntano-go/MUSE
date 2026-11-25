"""Core ACE Framework orchestrator"""
import logging
import json
import platform
import hashlib
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, List
from pathlib import Path

from ..config import Config
from ..models.playbook import Playbook
from ..models.skill_library import SkillLibrary
from ..agents.generator import Generator
from ..agents.reflector import Reflector
from ..agents.curator import Curator
from ..agents.skill_generator import SkillGenerator
from ..utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class ACEFramework:
    """Main ACE framework orchestrator"""
    
    def __init__(self, config: Config = None):
        """
        Initialize ACE Framework
        
        Args:
            config: Configuration object (uses default if None)
        """
        if config is None:
            raise ValueError("ACEFramework requires an explicit Config. Set experiment parameters in main.py.")
        self.config = config
        self.run_started_at_utc = datetime.now(timezone.utc).isoformat()
        
        # Initialize components
        self.playbook = Playbook()
        self.skill_library = SkillLibrary()
        self.reflexion_memory: Dict[str, str] = {}
        self.recent_trajectories: List[Dict[str, Any]] = []
        self.lifecycle_history: List[Dict[str, Any]] = []
        self.distillation_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.reflection_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.reflection_calls = 0
        self.curation_calls = 0
        self.run_status = "running"
        self.metrics = MetricsTracker()
        
        # Initialize agents
        self.generator = Generator(self.config.model)
        self.reflector = Reflector(self.config.model)
        self.curator = Curator(
            self.config.model,
            self.config.prompt.curator_sections
        )
        self.skill_generator = SkillGenerator(self.config.model)
        
        logger.info("ACE Framework initialized")

    def process_batch(
        self,
        samples: List[Dict[str, Any]],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process samples sequentially so each answer is recorded before its label
        can affect later samples in a prequential run.
        """
        if not samples:
            return []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing batch: {len(samples)} samples sequentially")

        batch_results = []
        for index, sample in enumerate(samples):
            try:
                result = self.predict_sample(sample, verbose=verbose, index=index)
            except Exception as exc:
                logger.exception("Sample %s failed", index)
                result = self._failed_sample(sample, str(exc))
            self.observe_sample(result, verbose=verbose)
            batch_results.append(result)

        if verbose:
            print(f"  Batch complete. Accuracy: {self.metrics.get_accuracy():.1f}%")

        return batch_results

    def predict_sample(
        self,
        sample: Dict[str, Any],
        verbose: bool = False,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Generate one answer without scoring it or applying feedback updates."""
        return self._generate_and_reflect_sample(index, sample, verbose)

    def observe_sample(self, result: Dict[str, Any], verbose: bool = False) -> None:
        """Reveal one gold label, score its prediction, then apply method updates."""
        self._after_sample(result, verbose=verbose)

    @staticmethod
    def _failed_sample(sample: Dict[str, Any], error: str) -> Dict[str, Any]:
        return {
            "sample": sample,
            "generation": {
                "reasoning": "",
                "used_bullet_ids": [],
                "used_skill_ids": [],
                "final_answer": "",
                "_usage": {},
                "_finish_reason": "request_failed",
                "_response_chars": 0,
                "_raw_content": "",
            },
            "reflection": {
                "error_identification": "Request failed",
                "root_cause": error,
                "key_insight": "",
                "bullet_tags": [],
                "skill_tags": [],
                "is_correct": False,
            },
            "correct": False,
            "parse_failed": True,
            "retrieved_skill_ids": [],
            "activated_skill_ids": [],
        }

    def _after_sample(self, result: Dict[str, Any], verbose: bool = False) -> None:
        """Record the prediction before revealing its label and updating state."""
        sample = result["sample"]
        generation = result["generation"]
        reflection = result["reflection"]
        line = self.config.experiment.line

        self.metrics.record_result(
            question=sample["question"],
            sample_id=sample.get("sample_id", ""),
            context=sample.get("context", ""),
            reasoning=generation.get("reasoning", ""),
            predicted=generation.get("final_answer", ""),
            ground_truth="",
            is_correct=False,
            playbook_size=len(self.playbook),
            used_skill_ids=generation.get("used_skill_ids", []),
            retrieved_skill_ids=result.get("retrieved_skill_ids", []),
            activated_skill_ids=result.get("activated_skill_ids", []),
            finish_reason=generation.get("_finish_reason") or "unknown",
            parse_failed=bool(result.get("parse_failed", False)),
            truncated=bool(result.get("truncated", generation.get("_finish_reason") == "length")),
            usage=generation.get("_usage", {}),
            reflection_usage=result.get("_reflection_usage", {}),
            source_index=sample.get("source_index"),
            raw_completion=generation.get("_raw_content", ""),
        )
        record = self.metrics.history[-1]
        scores = self.reflector.score_answer_metrics(
            generation.get("final_answer", ""),
            sample["answer"],
            self.config.experiment.metric_name
            if self.config.experiment.metric_name != "dataset_default"
            else self.config.experiment.dataset_name,
        )
        is_correct = scores["exact_match"]
        record["ground_truth"] = sample["answer"]
        record["correct"] = is_correct
        record["exact_match"] = is_correct
        record["f1"] = scores["f1"]
        self.metrics.record_f1(scores["f1"])
        self.metrics.correct += int(is_correct)
        record["accuracy"] = self.metrics.get_accuracy()
        reflection["is_correct"] = is_correct
        result["correct"] = is_correct

        is_frozen = (
            self.config.experiment.eval_protocol == "frozen"
            or self.config.experiment.freeze_transferred_skills
        )
        generation_failed = (
            generation.get("_finish_reason") in {"request_failed", "request_failed_or_unparseable"}
            and not generation.get("_raw_content")
        )
        if not is_frozen and not generation_failed and line in {"skillline", "baseline", "reflexion"}:
            reflection = self.reflector.reflect(
                sample["question"],
                generation.get("reasoning", ""),
                generation.get("final_answer", ""),
                sample["answer"],
                generation.get("used_bullet_ids", []),
                self.playbook,
                used_skills=generation.get("used_skill_ids", []),
                dataset_name=self.config.experiment.dataset_name,
                metric_name=self.config.experiment.metric_name,
            )
            reflection_usage = dict(self.reflector.last_usage)
            self._add_usage(self.reflection_usage, reflection_usage)
            self.reflection_calls += 1
            self.metrics.record_auxiliary_usage(reflection_usage)
            result["reflection"] = reflection
            result["_reflection_usage"] = reflection_usage
        else:
            result["_reflection_usage"] = {}

        if not is_frozen and not generation_failed and line == "skillline":
            self.skill_library.update_usage(generation.get("used_skill_ids", []))
            for tag in reflection.get("skill_tags", []):
                skill_id = tag.get("id")
                if skill_id:
                    self.skill_library.update_feedback(skill_id, tag.get("tag") == "helpful")
        elif not is_frozen and not generation_failed and line == "baseline" and not self.config.experiment.disable_playbook_updates:
            for tag in reflection.get("bullet_tags", []):
                bullet_id = tag.get("id")
                if bullet_id:
                    self.playbook.update_feedback(bullet_id, tag.get("tag") == "helpful")
            operations = self.curator.curate(reflection, self.playbook)
            self._add_usage(self.reflection_usage, self.curator.last_usage)
            self.curation_calls += 1
            self.metrics.record_auxiliary_usage(self.curator.last_usage)
            for operation in operations:
                if operation.get("type") == "ADD":
                    self.playbook.add_bullet(operation["section"], operation["content"])
                    self.metrics.record_playbook_update()
        elif not is_frozen and not generation_failed and line == "reflexion" and not reflection.get("is_correct", False):
            self.reflexion_memory[sample["question"]] = self._format_reflexion_memory(reflection)

        trajectory = {
            "sample": {"question": sample["question"], "context": sample.get("context", "")},
            "generation": {
                "final_answer": generation.get("final_answer", ""),
                "reasoning": generation.get("reasoning", ""),
            },
            "feedback": {
                "correct": is_correct,
                "key_insight": reflection.get("key_insight", ""),
                "root_cause": reflection.get("root_cause", ""),
            },
        }
        if generation.get("_finish_reason") == "length":
            result["truncated"] = True
        if not is_frozen and not generation_failed and line == "skillline":
            self.recent_trajectories.append(trajectory)
            del self.recent_trajectories[:-self.config.experiment.trajectory_window_size]

        if not generation_failed and (
            line == "skillline"
            and self.config.experiment.eval_protocol == "prequential"
            and self.metrics.processed % self.config.experiment.skill_update_interval == 0
        ):
            self.update_skills(verbose=verbose)

    def _generate_and_reflect_sample(
        self,
        index: int,
        sample: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Generate and reflect for one sample without updating playbook."""
        line = self.config.experiment.line
        use_skills = line == "skillline"
        generation = self.generator.generate(
            sample["question"],
            sample["context"],
            self.playbook,
            self.skill_library if use_skills else None,
            line=line,
            reflexion_memory=self.reflexion_memory.get(sample["question"], ""),
            skill_top_k=self.config.experiment.skill_top_k,
            dataset_name=self.config.experiment.dataset_name,
            retrieval_pool_size=self.config.experiment.retrieval_pool_size,
            use_uncertainty_gate=self.config.experiment.use_uncertainty_gate,
            gate_threshold=self.config.experiment.gate_threshold,
            gate_uncertainty_adjustment=self.config.experiment.gate_uncertainty_adjustment,
            gate_helpfulness_adjustment=self.config.experiment.gate_helpfulness_adjustment,
            gate_alpha=self.config.experiment.gate_alpha,
            gate_beta=self.config.experiment.gate_beta,
            gate_gamma=self.config.experiment.gate_gamma,
            gate_delta=self.config.experiment.gate_delta,
            record_skill_statistics=(
                self.config.experiment.eval_protocol != "frozen"
                and not self.config.experiment.freeze_transferred_skills
            ),
        )
        retrieved_skill_ids = list(self.generator.last_retrieved_skill_ids)
        activated_skill_ids = list(self.generator.last_activated_skill_ids)
        reflection = {
            "error_identification": "",
            "root_cause": "",
            "key_insight": "",
            "bullet_tags": [],
            "skill_tags": [],
            "is_correct": False,
        }

        return {
            "sample": sample,
            "generation": generation,
            "reflection": reflection,
            "correct": False,
            "skills": self.skill_library.get_statistics(),
            "retrieved_skill_ids": retrieved_skill_ids,
            "activated_skill_ids": activated_skill_ids,
            "parse_failed": not bool(generation.get("final_answer")),
            "truncated": generation.get("_finish_reason") == "length",
            "_reflection_usage": {},
        }

    @staticmethod
    def _add_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = total.get(key, 0) + int((usage or {}).get(key, 0) or 0)

    @staticmethod
    def _combine_batch_reflections(reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple sample reflections into one batch-level reflection for playbook curation."""
        if not reflections:
            return {
                "error_identification": "",
                "root_cause": "",
                "key_insight": "",
                "bullet_tags": [],
                "is_correct": True
            }

        incorrect_reflections = [r for r in reflections if not r.get("is_correct", False)]
        source_reflections = incorrect_reflections or reflections

        return {
            "error_identification": "\n".join(
                f"- {r.get('error_identification', '')}" for r in source_reflections if r.get("error_identification")
            ),
            "root_cause": "\n".join(
                f"- {r.get('root_cause', '')}" for r in source_reflections if r.get("root_cause")
            ),
            "key_insight": "\n".join(
                f"- {r.get('key_insight', '')}" for r in source_reflections if r.get("key_insight")
            ),
            "bullet_tags": [
                tag_info
                for reflection in reflections
                for tag_info in reflection.get("bullet_tags", [])
            ],
            "is_correct": all(r.get("is_correct", False) for r in reflections)
        }
    
    def adapt_online(
        self,
        question: str,
        context: str,
        ground_truth: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Process one sample through the same protocol as process_batch.
        
        Args:
            question: Question to answer
            context: Supporting context
            ground_truth: Correct answer
            verbose: Whether to print progress
            
        Returns:
            Dictionary with generation, reflection, and correctness
        """
        sample = {"question": question, "context": context, "answer": ground_truth}
        result = self._generate_and_reflect_sample(0, sample, verbose)
        self._after_sample(result, verbose=verbose)
        if verbose:
            if result["correct"]:
                print(f"  Correct! Accuracy: {self.metrics.get_accuracy():.1f}%")
            else:
                print(f"   Incorrect. Accuracy: {self.metrics.get_accuracy():.1f}%")
        return result

    def update_skills(self, verbose: bool = True) -> None:
        """Generate/update skills from the current playbook"""
        if self.config.experiment.freeze_transferred_skills:
            if verbose:
                print("   Skill update skipped: transferred skills are frozen")
            return

        if self.config.experiment.line == "skillline":
            window_size = self.config.experiment.trajectory_window_size
            max_skills = max(1, min(8, window_size))
            generated_skills = self.skill_generator.generate_macro_skills(
                self.recent_trajectories[-window_size:],
                dataset_name=self.config.experiment.dataset_name,
                max_skills=max_skills
            )
        else:
            max_skills = max(3, min(8, len(self.playbook)))
            generated_skills = self.skill_generator.generate_skills(self.playbook, max_skills=max_skills)
        self._add_usage(self.distillation_usage, self.skill_generator.last_usage)
        if generated_skills:
            exp = self.config.experiment
            if exp.use_lifecycle_management:
                lifecycle = self.skill_library.update_library(
                    generated_skills,
                    max_library_size=exp.max_library_size,
                    retain_threshold=exp.retain_threshold,
                    retire_threshold=exp.retire_threshold,
                    utility_weights=(
                        exp.utility_confidence_weight,
                        exp.utility_helpfulness_weight,
                        exp.utility_usage_weight,
                        exp.utility_redundancy_weight,
                    ),
                )
            else:
                self.skill_library.replace_skills(generated_skills)
                lifecycle = {
                    "candidate_count": len(generated_skills),
                    "retained_count": len(self.skill_library),
                    "retired_count": 0,
                    "retained_ids": [skill.id for skill in self.skill_library.skills],
                    "utility_scores": {},
                }
            lifecycle["step"] = self.metrics.processed
            self.lifecycle_history.append(lifecycle)
            if verbose:
                print(f"   Skill library update at step {self.metrics.processed}: {len(self.skill_library)} retained")
        else:
            self.lifecycle_history.append({
                "step": self.metrics.processed,
                "candidate_count": 0,
                "retained_count": len(self.skill_library),
                "retired_count": 0,
                "retained_ids": [skill.id for skill in self.skill_library.skills],
                "utility_scores": {},
                "generation_empty": True,
            })
            if verbose:
                print("   Skill update skipped: no valid skills generated")

    @staticmethod
    def _format_reflexion_memory(reflection: Dict[str, Any]) -> str:
        """Format reflection output as per-question Reflexion memory"""
        return (
            f"Previous error: {reflection.get('error_identification', '')}\n"
            f"Root cause: {reflection.get('root_cause', '')}\n"
            f"Lesson for retry: {reflection.get('key_insight', '')}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "metrics": self.metrics.get_summary(),
            "playbook": self.playbook.get_statistics(),
            "skills": self.skill_library.get_statistics(),
            "reflexion_memory_size": len(self.reflexion_memory)
        }
    
    def save_results(self, output_dir: str = None) -> None:
        """
        Save all results (playbook, metrics, plots)
        
        Args:
            output_dir: Output directory (uses config default if None)
        """
        if output_dir is None:
            output_dir = self.config.experiment.output_dir

        output_path = Path(output_dir)
        if output_path.exists() and any(output_path.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty experiment results directory: {output_path}"
            )

        if (
            self.config.experiment.line == "skillline"
            and self.config.experiment.eval_protocol == "prequential"
            and self.config.experiment.flush_final_skill_update
            and self.recent_trajectories
            and (not self.lifecycle_history or self.lifecycle_history[-1].get("step") != self.metrics.processed)
        ):
            self.update_skills(verbose=False)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save playbook
        playbook_path = output_path / self.config.experiment.playbook_filename
        self.playbook.save(str(playbook_path))
        logger.info(f"Playbook saved to {playbook_path}")

        # Save skills
        skills_path = output_path / self.config.experiment.skills_filename
        self.skill_library.save(str(skills_path))
        logger.info(f"Skills saved to {skills_path}")
        
        # Save metrics
        metrics_path = output_path / self.config.experiment.metrics_filename
        self.metrics.save(str(metrics_path))
        logger.info(f"Metrics saved to {metrics_path}")

        metadata_path = output_path / "run_metadata.json"
        metadata = {
            "config": self.config.to_dict(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "api_base_url": self.config.model.base_url,
            "api_model_id": self.config.model.name,
            "api_snapshot": None,
            "api_snapshot_note": "Provider snapshot/version was not supplied by the configured endpoint.",
            "run_started_at_utc": self.run_started_at_utc,
            "run_saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_source_sha256": {
                name: self._file_sha256(path)
                for name, path in {
                    "generator": Path(__file__).parents[1] / "agents" / "generator.py",
                    "reflector": Path(__file__).parents[1] / "agents" / "reflector.py",
                    "skill_generator": Path(__file__).parents[1] / "agents" / "skill_generator.py",
                }.items()
            },
            "run_status": self.run_status,
            "dataset_source_path": self.config.experiment.dataset_path,
            "dataset_source_sha256": self._file_sha256(self.config.experiment.dataset_path),
            "evaluation_protocol": self.config.experiment.eval_protocol,
            "processed_sample_ids": [row.get("sample_id", "") for row in self.metrics.history],
            "processed_source_indices": [row.get("source_index") for row in self.metrics.history],
            "distillation_usage": self.distillation_usage,
            "reflection_and_curation_usage": self.reflection_usage,
            "reflection_calls": self.reflection_calls,
            "curation_calls": self.curation_calls,
            "solver_token_totals": {
                "prompt_tokens": self.metrics.generator_prompt_tokens,
                "completion_tokens": self.metrics.generator_completion_tokens,
            },
            "measured_total_token_totals": {
                "prompt_tokens": self.metrics.generator_prompt_tokens + self.reflection_usage["prompt_tokens"] + self.distillation_usage["prompt_tokens"],
                "completion_tokens": self.metrics.generator_completion_tokens + self.reflection_usage["completion_tokens"] + self.distillation_usage["completion_tokens"],
            },
            "measured_total_tokens_amortized_per_processed_task": (
                self.metrics.generator_prompt_tokens
                + self.metrics.generator_completion_tokens
                + self.reflection_usage["prompt_tokens"]
                + self.reflection_usage["completion_tokens"]
                + self.distillation_usage["prompt_tokens"]
                + self.distillation_usage["completion_tokens"]
            ) / self.metrics.processed if self.metrics.processed else 0.0,
            "lifecycle_history": self.lifecycle_history,
            "skill_library_final": self.skill_library.get_statistics(),
            "skill_library_final_size": len(self.skill_library),
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        
        # Save tabular results
        table_path = output_path / self.config.experiment.results_table_filename
        self.metrics.save_table(str(table_path))
        logger.info(f"Results table saved to {table_path}")
        
        logger.info(f"All results saved to {output_dir}")

    @staticmethod
    def _file_sha256(filepath: str) -> str:
        path = Path(filepath)
        if not path.is_file():
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    
    def save_checkpoint(self, checkpoint_dir: str, runner_state: Dict[str, Any]) -> Path:
        """Persist the complete online learner state into a new snapshot directory."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mkdtemp(prefix=f".{checkpoint_path.name}.tmp-", dir=checkpoint_path.parent))
        try:
            self.playbook.save(str(temp_path / self.config.experiment.playbook_filename))
            self.skill_library.save(str(temp_path / self.config.experiment.skills_filename))
            self.metrics.save(str(temp_path / self.config.experiment.metrics_filename))
            state = {
            "config_fingerprint": self.config.to_dict(),
                "run_status": self.run_status,
                "run_started_at_utc": self.run_started_at_utc,
                "reflexion_memory": self.reflexion_memory,
                "recent_trajectories": self.recent_trajectories,
                "lifecycle_history": self.lifecycle_history,
                "distillation_usage": self.distillation_usage,
                "reflection_usage": self.reflection_usage,
                "reflection_calls": self.reflection_calls,
                "curation_calls": self.curation_calls,
                "runner": runner_state,
            }
            with (temp_path / "online_state.json").open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, ensure_ascii=False)
            if checkpoint_path.exists():
                raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_path}")
            temp_path.rename(checkpoint_path)
            return checkpoint_path.resolve()
        except Exception:
            for path in temp_path.iterdir():
                path.unlink()
            temp_path.rmdir()
            raise

    def load_checkpoint(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Load checkpoint from directory
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
        """
        checkpoint_path = Path(checkpoint_dir)
        
        # Load playbook
        playbook_path = checkpoint_path / self.config.experiment.playbook_filename
        if playbook_path.exists():
            self.playbook.load(str(playbook_path))
            logger.info(f"Loaded playbook from {playbook_path}")

        # Load skills
        skills_path = checkpoint_path / self.config.experiment.skills_filename
        if skills_path.exists():
            self.skill_library.load(str(skills_path))
            logger.info(f"Loaded skills from {skills_path}")
        
        # Load metrics
        metrics_path = checkpoint_path / self.config.experiment.metrics_filename
        if metrics_path.exists():
            self.metrics.load(str(metrics_path))
            logger.info(f"Loaded metrics from {metrics_path}")

        state_path = checkpoint_path / "online_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"Online checkpoint state is missing: {state_path}")
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("config_fingerprint") != self.config.to_dict():
            raise ValueError(f"Checkpoint configuration does not match: {checkpoint_path}")
        self.run_status = state.get("run_status", "running")
        self.run_started_at_utc = state.get("run_started_at_utc", self.run_started_at_utc)
        self.reflexion_memory = state.get("reflexion_memory", {})
        self.recent_trajectories = state.get("recent_trajectories", [])
        self.lifecycle_history = state.get("lifecycle_history", [])
        self.distillation_usage = state.get("distillation_usage", self.distillation_usage)
        self.reflection_usage = state.get("reflection_usage", self.reflection_usage)
        self.reflection_calls = int(state.get("reflection_calls", 0))
        self.curation_calls = int(state.get("curation_calls", 0))
        self.metrics.processed = len(self.metrics.history)
        self.metrics.correct = sum(bool(row.get("correct")) for row in self.metrics.history)
        f1_values = [row.get("f1") for row in self.metrics.history if row.get("f1") is not None]
        self.metrics.f1_count = len(f1_values)
        self.metrics.f1_total = sum(float(value) for value in f1_values)
        self.metrics.total_prompt_tokens = sum(
            int(row.get("total_prompt_tokens", 0) or 0) for row in self.metrics.history
        )
        self.metrics.total_completion_tokens = sum(
            int(row.get("total_completion_tokens", 0) or 0) for row in self.metrics.history
        )
        self.metrics.generator_prompt_tokens = sum(
            int(row.get("prompt_tokens", 0) or 0) for row in self.metrics.history
        )
        self.metrics.generator_completion_tokens = sum(
            int(row.get("completion_tokens", 0) or 0) for row in self.metrics.history
        )
        self.metrics.reflection_prompt_tokens = sum(
            int(row.get("reflection_prompt_tokens", 0) or 0) for row in self.metrics.history
        )
        self.metrics.reflection_completion_tokens = sum(
            int(row.get("reflection_completion_tokens", 0) or 0) for row in self.metrics.history
        )
        return state.get("runner", {})
    
    def print_summary(self) -> None:
        """Print execution summary"""
        print("\n" + "="*60)
        print(" ACE Adaptation Complete!")
        print(f"   Final Accuracy: {self.metrics.get_accuracy():.1f}%")
        if self.metrics.f1_count:
            print(f"   HotpotQA F1: {self.metrics.f1_total / self.metrics.f1_count:.4f}")
        print(f"   Questions Processed: {self.metrics.processed}")
        print(f"   Correct Answers: {self.metrics.correct}")
        print(f"   Playbook Size: {len(self.playbook)} strategies")
        print(f"   Skill Library Size: {len(self.skill_library)} skills")
        print(f"   Reflexion Memory Size: {len(self.reflexion_memory)} items")
        print(f"   Total Updates: {self.metrics.playbook_updates}")
        
        # Print sample strategies
        if len(self.playbook) > 0:
            print("\n Sample Strategies from Playbook:")
            for i, bullet in enumerate(self.playbook.bullets[:5]):
                print(f"   [{bullet.id}] {bullet.content}")
                if i >= 4:
                    break
        
        print("="*60)
