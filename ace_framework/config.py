"""Configuration management for ACE Framework"""
import os
from dataclasses import dataclass, asdict
from typing import Optional
from dotenv import load_dotenv


load_dotenv()


@dataclass
class ModelConfig:
    """LLM model configuration"""
    name: str
    temperature: float
    max_tokens: int
    enable_thinking: bool
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not self.api_key:
                raise ValueError("DASHSCOPE_API_KEY not found in environment")
        if self.base_url is None:
            self.base_url = os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    dataset_name: str
    dataset_path: str
    dataset_split: str
    num_samples: int
    random_seed: int
    experiment_rounds: int
    checkpoint_interval: int
    line: str
    skill_update_interval: int
    skill_top_k: int
    batch_parallel_size: int
    output_dir: str
    eval_protocol: str = "prequential"
    flush_final_skill_update: bool = True
    trajectory_window_size: int = 20
    retrieval_pool_size: int = 20
    use_uncertainty_gate: bool = True
    gate_alpha: float = 0.45
    gate_beta: float = 0.25
    gate_gamma: float = 0.20
    gate_delta: float = 0.10
    gate_threshold: float = 0.45
    gate_uncertainty_adjustment: float = 0.10
    gate_helpfulness_adjustment: float = 0.05
    use_lifecycle_management: bool = True
    utility_confidence_weight: float = 0.50
    utility_helpfulness_weight: float = 0.25
    utility_usage_weight: float = 0.05
    utility_redundancy_weight: float = 0.20
    retain_threshold: float = 0.20
    retire_threshold: float = 0.05
    max_library_size: int = 32
    metric_name: str = "dataset_default"
    transfer_skill_path: Optional[str] = None
    transfer_playbook_path: Optional[str] = None
    transfer_artifact_sha256: Optional[str] = None
    freeze_transferred_skills: bool = False
    disable_playbook_updates: bool = False
    playbook_filename: str = "ace_playbook.json"
    skills_filename: str = "ace_skills.json"
    metrics_filename: str = "metrics.json"
    results_table_filename: str = "results_table.csv"
    
    def __post_init__(self):
        valid_lines = {"base_llm", "cot", "reflexion", "baseline", "skillline"}
        if self.line not in valid_lines:
            raise ValueError(f"experiment line must be one of {sorted(valid_lines)}")
        if not self.dataset_split.strip():
            raise ValueError("dataset_split must identify the source split or local file")
        if self.experiment_rounds < 1:
            raise ValueError("experiment_rounds must be >= 1")
        if self.experiment_rounds != 1:
            raise ValueError(
                "Repeated evaluation rounds reuse labeled samples and can leak answers; "
                "use a separate frozen evaluation instead."
            )
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.skill_update_interval < 1:
            raise ValueError("skill_update_interval must be >= 1")
        if self.skill_top_k < 1:
            raise ValueError("skill_top_k must be >= 1")
        if self.batch_parallel_size < 1:
            raise ValueError("batch_parallel_size must be >= 1")
        if self.trajectory_window_size < 1:
            raise ValueError("trajectory_window_size must be >= 1")
        if self.retrieval_pool_size < self.skill_top_k:
            raise ValueError("retrieval_pool_size must be >= skill_top_k")
        if self.max_library_size < 1:
            raise ValueError("max_library_size must be >= 1")
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be between 0 and 1")
        if any(not 0.0 <= value <= 1.0 for value in (
            self.gate_alpha, self.gate_beta, self.gate_gamma, self.gate_delta,
            self.gate_uncertainty_adjustment, self.gate_helpfulness_adjustment,
            self.retain_threshold, self.retire_threshold,
        )):
            raise ValueError("gate and lifecycle scores must be between 0 and 1")
        if any(value < 0.0 for value in (
            self.utility_confidence_weight,
            self.utility_helpfulness_weight,
            self.utility_usage_weight,
            self.utility_redundancy_weight,
        )):
            raise ValueError("utility weights must be non-negative")
        if self.retire_threshold > self.retain_threshold:
            raise ValueError("retire_threshold must be <= retain_threshold")
        if self.eval_protocol not in {"prequential", "frozen"}:
            raise ValueError("eval_protocol must be 'prequential' or 'frozen'")
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class PromptConfig:
    """Prompt templates configuration"""
    generator_sections: list = None
    reflector_sections: list = None
    curator_sections: list = None
    
    def __post_init__(self):
        if self.generator_sections is None:
            self.generator_sections = [
                "multi_hop_reasoning",
                "entity_extraction",
                "fact_verification",
                "common_mistakes"
            ]
        if self.reflector_sections is None:
            self.reflector_sections = self.generator_sections
        if self.curator_sections is None:
            self.curator_sections = self.generator_sections


class Config:
    """Main configuration class"""
    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
        prompt_config: Optional[PromptConfig] = None
    ):
        self.model = model_config
        self.experiment = experiment_config
        self.prompt = prompt_config or PromptConfig()

    def to_dict(self) -> dict:
        """Return a JSON-serializable configuration without credentials."""
        model = asdict(self.model)
        model.pop("api_key", None)
        model["base_url"] = self.model.base_url
        return {
            "model": model,
            "experiment": asdict(self.experiment),
            "prompt": asdict(self.prompt),
        }
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration"""
        raise ValueError("Config.default() is disabled. Set experiment parameters explicitly in main.py.")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from dictionary"""
        return cls(
            model_config=ModelConfig(**config_dict.get("model", {})),
            experiment_config=ExperimentConfig(**config_dict.get("experiment", {})),
            prompt_config=PromptConfig(**config_dict.get("prompt", {}))
        )
