"""Dataset loading and preprocessing"""
import json
import random
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional
from datasets import load_from_disk, Dataset, concatenate_datasets
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and preprocessing of datasets"""
    
    def __init__(self, dataset_path: str, dataset_name: str = "hotpotqa", random_seed: int = 42):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name.lower()
        self.random_seed = random_seed
        self.dataset: Any = None
    
    def load(self) -> Dataset:
        """Load dataset from disk"""
        dataset_path = Path(self.dataset_path)

        preferred_jsonl = dataset_path.with_suffix(".jsonl")
        if dataset_path.suffix == ".json" and preferred_jsonl.exists():
            dataset_path = preferred_jsonl
            self.dataset_path = str(preferred_jsonl)

        if dataset_path.suffix == ".jsonl":
            with dataset_path.open("r", encoding="utf-8") as f:
                self.dataset = [json.loads(line) for line in f if line.strip()]
            logger.info(f"Loaded JSONL dataset with {len(self.dataset)} samples")
            return self.dataset

        if dataset_path.suffix == ".json":
            with dataset_path.open("r", encoding="utf-8") as f:
                try:
                    self.dataset = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    self.dataset = [json.loads(line) for line in f if line.strip()]
            logger.info(f"Loaded JSON dataset with {len(self.dataset)} samples")
            return self.dataset

        try:
            self.dataset = load_from_disk(self.dataset_path)
            logger.info(f"Loaded dataset with {len(self.dataset)} samples")
            return self.dataset
        except Exception as e:
            logger.warning(f"Failed to load dataset from disk format: {e}")

        arrow_files = []
        if dataset_path.is_file() and dataset_path.suffix == ".arrow":
            arrow_files = [dataset_path]
        elif dataset_path.is_dir():
            arrow_files = sorted(dataset_path.glob("*.arrow"))

        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {self.dataset_path}")

        datasets = [Dataset.from_file(str(arrow_file)) for arrow_file in arrow_files]
        self.dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
        logger.info(f"Loaded dataset with {len(self.dataset)} samples from Arrow file(s)")
        return self.dataset
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Get a single preprocessed sample
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with question, context, and answer
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        sample = self.dataset[index]
        return self.preprocess_sample(sample)
    
    def iterate_samples(self, num_samples: int = None) -> Iterator[Dict[str, Any]]:
        """
        Iterate over samples
        
        Args:
            num_samples: Number of samples to iterate (None for all)
            
        Yields:
            Preprocessed sample dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        total_samples = len(self.dataset)
        if num_samples is None:
            num_samples = total_samples
        else:
            num_samples = min(num_samples, total_samples)
        
        indices = list(range(total_samples))
        if num_samples is not None and num_samples < total_samples:
            rng = random.Random(self.random_seed)
            indices = rng.sample(indices, num_samples)
        else:
            indices = indices[:num_samples]

        for i in indices:
            sample = self.get_sample(i)
            sample["source_index"] = i
            if not sample.get("sample_id"):
                sample["sample_id"] = str(i)
            yield sample
    
    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a raw sample
        
        Args:
            sample: Raw sample from dataset
            
        Returns:
            Dictionary with question, context, and answer
        """
        if self.dataset_name in {"hotpotqa", "hotpot_dev_distractor_v1"}:
            return self._preprocess_hotpot(sample)
        if self.dataset_name == "gsm8k":
            return self._preprocess_gsm8k(sample)
        if self.dataset_name in {"gsm-hard", "gsm_hard", "gsmhard"}:
            return self._preprocess_gsm_hard(sample)
        if self.dataset_name == "math":
            return self._preprocess_math(sample)

        if {"question", "context", "answer"}.issubset(sample.keys()):
            return {
                "question": sample.get("question", ""),
                "context": sample.get("context", ""),
                "answer": sample.get("answer", ""),
                "sample_id": str(sample.get("_id", sample.get("idx", "")))
            }

        raise ValueError(f"Unsupported dataset format for dataset_name={self.dataset_name}")

    @staticmethod
    def _preprocess_hotpot(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess HotpotQA-style sample"""
        raw_context = sample.get("context", [])
        context_parts: List[str] = []

        if isinstance(raw_context, dict):
            context_sentences = raw_context.get("sentences", [])
            context_parts = [" ".join(ctx) for ctx in context_sentences]
        elif isinstance(raw_context, list):
            for item in raw_context:
                if isinstance(item, list) and len(item) >= 2:
                    title = item[0]
                    sentences = item[1]
                    if isinstance(sentences, list):
                        context_parts.append(f"{title}: {' '.join(sentences)}")
                else:
                    context_parts.append(str(item))

        return {
            "question": sample.get("question", ""),
            "context": " ".join(context_parts),
            "answer": sample.get("answer", ""),
            "sample_id": str(sample.get("_id", sample.get("idx", ""))),
        }

    @staticmethod
    def _extract_final_answer(answer: str) -> str:
        """Extract final short answer from chain-of-thought style solution"""
        if "####" in answer:
            return answer.split("####")[-1].strip()
        boxed = DatasetLoader._extract_boxed_answer(answer)
        if boxed is not None:
            return boxed
        return answer.strip()

    @staticmethod
    def _extract_boxed_answer(text: str) -> Optional[str]:
        import re
        matches = list(re.finditer(r"\\boxed\s*\{", text or ""))
        if not matches:
            return None
        start = matches[-1].end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip()
        return None

    def _preprocess_gsm8k(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess GSM8K sample"""
        return {
            "question": sample.get("question", ""),
            "context": "",
            "answer": self._extract_final_answer(sample.get("answer", "")),
            "sample_id": str(sample.get("idx", ""))
        }

    @staticmethod
    def _format_numeric_answer(answer: Any) -> str:
        """Format GSM-Hard numeric targets without unnecessary trailing .0."""
        if isinstance(answer, float) and answer.is_integer():
            return str(int(answer))
        return str(answer).strip()

    def _preprocess_gsm_hard(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess GSM-Hard sample with input/code/target fields."""
        return {
            "question": sample.get("input", sample.get("question", "")),
            "context": "",
            "answer": self._format_numeric_answer(sample.get("target", sample.get("answer", ""))),
            "sample_id": str(sample.get("idx", ""))
        }

    def _preprocess_math(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess MATH sample"""
        solution = sample.get("solution", "")
        return {
            "question": sample.get("problem", sample.get("question", "")),
            "context": f"Level: {sample.get('level', '')}; Type: {sample.get('type', '')}",
            "answer": self._extract_final_answer(solution),
            "sample_id": str(sample.get("idx", ""))
        }
    
    def __len__(self) -> int:
        """Get dataset size"""
        if self.dataset is None:
            return 0
        return len(self.dataset)
