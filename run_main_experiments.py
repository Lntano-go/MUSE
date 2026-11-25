"""Run matched online prequential experiments on one shared task stream."""
import argparse
import csv
import hashlib
import json
import logging
import random
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from ace_framework.config import Config, ExperimentConfig, ModelConfig
from ace_framework.core.ace_framework import ACEFramework
from ace_framework.data.dataset_loader import DatasetLoader


csv.field_size_limit(10 * 1024 * 1024)


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "hotpotqa": {
        "name": "hotpot_dev_distractor_v1",
        "stream": Path("hotpot_dev_distractor_v1/test.jsonl"),
        "split": "local_test_file_hotpot_dev_distractor_online_stream",
        "source_kind": "local labeled test/dev file; each label is revealed only after its prediction",
    },
    "math": {
        "name": "math",
        "stream": Path("math/test.jsonl"),
        "split": "test_online_stream",
        "source_kind": "local labeled test file; each label is revealed only after its prediction",
    },
    "gsm-hard": {
        "name": "gsm-hard",
        "stream": Path("gsm-hard/train.json"),
        "split": "local_single_file_online_stream",
        "source_kind": "one local file; sampled items are consumed once in online order",
    },
}
METHODS = {
    "base_llm": "base_llm",
    "ACE": "baseline",
    "MUSE": "skillline",
}
STREAM_SIZE = 100
HOLDOUT_SIZE = 0
CHECKPOINT_INTERVAL = 10
SEGMENT_SIZE = 50
PROTOCOL_VERSION = "matched-online-experiments-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_source_hashes() -> dict:
    return {
        str(path): sha256(ROOT / path)
        for path in (
            Path("run_main_experiments.py"),
            Path("ace_framework/config.py"),
            Path("ace_framework/core/ace_framework.py"),
            Path("ace_framework/agents/generator.py"),
            Path("ace_framework/agents/reflector.py"),
            Path("ace_framework/agents/curator.py"),
            Path("ace_framework/agents/skill_generator.py"),
            Path("ace_framework/data/dataset_loader.py"),
            Path("ace_framework/models/skill_library.py"),
            Path("ace_framework/utils/metrics.py"),
        )
    }


def load_rows(path: Path, dataset_name: str) -> Tuple[DatasetLoader, List[dict]]:
    loader = DatasetLoader(str(path), dataset_name=dataset_name, random_seed=38)
    loader.load()
    return loader, [loader.dataset[index] for index in range(len(loader))]


def prepare_samples(loader: DatasetLoader, rows: List[dict], indices: List[int]) -> List[dict]:
    samples = []
    for source_index in indices:
        sample = loader.preprocess_sample(rows[source_index])
        sample["source_index"] = source_index
        if not sample.get("sample_id"):
            sample["sample_id"] = str(source_index)
        samples.append(sample)
    return samples


def question_hash(sample: dict) -> str:
    question = re.sub(r"\s+", " ", sample.get("question", "")).strip().lower()
    context = re.sub(r"\s+", " ", sample.get("context", "")).strip().lower()
    return hashlib.sha256(f"{question}\n{context}".encode("utf-8")).hexdigest()


def choose_stream(
    spec: dict,
    seed: int,
    count: int = STREAM_SIZE,
    excluded_indices: List[int] | None = None,
    excluded_question_hashes: set[str] | None = None,
) -> Tuple[List[dict], dict]:
    stream_path = ROOT / spec["stream"]
    loader, rows = load_rows(stream_path, spec["name"])
    excluded = set(excluded_indices or [])
    available = [index for index in range(len(rows)) if index not in excluded]
    samples = []
    used_ids = set()
    used_questions = set(excluded_question_hashes or set())
    candidates = list(available)
    random.Random(seed).shuffle(candidates)
    for source_index in candidates:
        sample = prepare_samples(loader, rows, [source_index])[0]
        normalized_question = question_hash(sample)
        if sample["sample_id"] in used_ids or normalized_question in used_questions:
            continue
        if not sample.get("answer"):
            raise ValueError(f"Missing gold answer at source index {source_index} in {spec['name']}")
        samples.append(sample)
        used_ids.add(sample["sample_id"])
        used_questions.add(normalized_question)
        if len(samples) == count:
            break
    if len(samples) < count:
        raise ValueError(
            f"{stream_path} has only {len(samples)} unique available samples after exclusions; "
            f"{count} requested"
        )

    return samples, {
        "dataset_name": spec["name"],
        "protocol": "One shared stream of unique items. Each method predicts item t, then receives its label and updates before item t+1.",
        "source_kind": spec["source_kind"],
        "stream_length": count,
        "stream_order": "seeded random sample without replacement",
        "source": {"path": str(stream_path.relative_to(ROOT)), "sha256": sha256(stream_path)},
        "stream": [
            {"sample_id": sample["sample_id"], "source_index": sample["source_index"]}
            for sample in samples
        ],
    }


def choose_splits(spec: dict, seed: int, stream_size: int, holdout_size: int) -> tuple[list[dict], dict, list[dict], dict | None]:
    train_samples, train_info = choose_stream(spec, seed, stream_size)
    if holdout_size == 0:
        return train_samples, train_info, [], None
    excluded_indices = [sample["source_index"] for sample in train_samples]
    excluded_questions = {question_hash(sample) for sample in train_samples}
    holdout_samples, holdout_info = choose_stream(
        spec,
        seed + 1,
        holdout_size,
        excluded_indices=excluded_indices,
        excluded_question_hashes=excluded_questions,
    )
    holdout_info["protocol"] = "Frozen holdout disjoint from the online stream; no labels are used for updates."
    return train_samples, train_info, holdout_samples, holdout_info


def samples_from_manifest(spec: dict, split_info: dict) -> list[dict]:
    stream_path = ROOT / spec["stream"]
    expected_source = split_info.get("source", {})
    if expected_source.get("sha256") != sha256(stream_path):
        raise ValueError(f"Dataset source changed since the sampling manifest was written: {stream_path}")
    loader, rows = load_rows(stream_path, spec["name"])
    samples = prepare_samples(
        loader,
        rows,
        [int(item["source_index"]) for item in split_info.get("stream", [])],
    )
    expected_ids = [item["sample_id"] for item in split_info.get("stream", [])]
    if [sample["sample_id"] for sample in samples] != expected_ids:
        raise ValueError(f"Sample IDs no longer match the sampling manifest for {spec['name']}")
    return samples


def make_config(
    model_name: str,
    dataset_name: str,
    dataset_path: Path,
    dataset_split: str,
    line: str,
    output_dir: Path,
    seed: int,
    num_samples: int,
    args=None,
    method_spec: dict | None = None,
    frozen: bool = False,
) -> Config:
    method_spec = method_spec or {}
    overrides = method_spec.get("overrides", {})
    artifact_path = method_spec.get("artifact_path")
    artifact_kind = method_spec.get("artifact_kind")
    artifact_file = Path(artifact_path).expanduser().resolve() if artifact_path else None
    if artifact_file and not artifact_file.is_file():
        raise FileNotFoundError(f"Transfer artifact not found: {artifact_file}")
    artifact_hash = sha256(artifact_file) if artifact_file else None
    experiment_options = {
        "skill_update_interval": getattr(args, "skill_update_interval", 100),
        "skill_top_k": getattr(args, "skill_top_k", 8),
        "retrieval_pool_size": getattr(args, "retrieval_pool_size", 20),
        "trajectory_window_size": getattr(args, "trajectory_window_size", 20),
        "gate_alpha": getattr(args, "gate_alpha", 0.45),
        "gate_beta": getattr(args, "gate_beta", 0.25),
        "gate_gamma": getattr(args, "gate_gamma", 0.20),
        "gate_delta": getattr(args, "gate_delta", 0.10),
        "gate_threshold": getattr(args, "gate_threshold", 0.45),
        "gate_uncertainty_adjustment": getattr(args, "gate_uncertainty_adjustment", 0.10),
        "gate_helpfulness_adjustment": getattr(args, "gate_helpfulness_adjustment", 0.05),
        "retain_threshold": getattr(args, "retain_threshold", 0.20),
        "retire_threshold": getattr(args, "retire_threshold", 0.05),
        "max_library_size": getattr(args, "max_library_size", 20),
        "utility_confidence_weight": getattr(args, "utility_confidence_weight", 0.50),
        "utility_helpfulness_weight": getattr(args, "utility_helpfulness_weight", 0.25),
        "utility_usage_weight": getattr(args, "utility_usage_weight", 0.05),
        "utility_redundancy_weight": getattr(args, "utility_redundancy_weight", 0.20),
    }
    experiment_options.update(overrides)
    is_frozen = frozen or bool(method_spec.get("frozen"))
    return Config(
        model_config=ModelConfig(
            name=model_name,
            temperature=getattr(args, "temperature", 0.3),
            max_tokens=getattr(args, "max_tokens", 8000),
            enable_thinking=getattr(args, "enable_thinking", False),
        ),
        experiment_config=ExperimentConfig(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            dataset_split=dataset_split,
            num_samples=num_samples,
            random_seed=seed,
            experiment_rounds=1,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            line=line,
            skill_update_interval=experiment_options["skill_update_interval"],
            flush_final_skill_update=False,
            skill_top_k=experiment_options["skill_top_k"],
            retrieval_pool_size=experiment_options["retrieval_pool_size"],
            trajectory_window_size=experiment_options["trajectory_window_size"],
            batch_parallel_size=getattr(args, "batch_parallel_size", 1),
            eval_protocol="frozen" if is_frozen else "prequential",
            use_uncertainty_gate=experiment_options.get("use_uncertainty_gate", True),
            gate_threshold=experiment_options["gate_threshold"],
            gate_alpha=experiment_options["gate_alpha"],
            gate_beta=experiment_options["gate_beta"],
            gate_gamma=experiment_options["gate_gamma"],
            gate_delta=experiment_options["gate_delta"],
            gate_uncertainty_adjustment=experiment_options["gate_uncertainty_adjustment"],
            gate_helpfulness_adjustment=experiment_options["gate_helpfulness_adjustment"],
            use_lifecycle_management=experiment_options.get("use_lifecycle_management", True),
            utility_confidence_weight=experiment_options["utility_confidence_weight"],
            utility_helpfulness_weight=experiment_options["utility_helpfulness_weight"],
            utility_usage_weight=experiment_options["utility_usage_weight"],
            utility_redundancy_weight=experiment_options["utility_redundancy_weight"],
            retain_threshold=experiment_options["retain_threshold"],
            retire_threshold=experiment_options["retire_threshold"],
            max_library_size=experiment_options["max_library_size"],
            transfer_skill_path=str(artifact_file) if artifact_kind == "skills" else None,
            transfer_playbook_path=str(artifact_file) if artifact_kind == "playbook" else None,
            transfer_artifact_sha256=artifact_hash,
            freeze_transferred_skills=is_frozen,
            disable_playbook_updates=is_frozen,
            output_dir=str(output_dir),
        ),
    )


def ensure_empty(path: Path) -> None:
    if path.exists() and any(item.is_file() for item in path.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {path}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def checkpoint_path(method_root: Path, processed: int) -> Path:
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return method_root / "checkpoints" / f"step_{processed:04d}_{suffix}"


def checkpoint_commit_path(dataset_root: Path, processed: int) -> Path:
    return dataset_root / "checkpoints" / f"step_{processed:04d}.complete.json"


def latest_checkpoint(method_root: Path) -> Path | None:
    dataset_root = method_root.parent
    checkpoint_root = dataset_root / "checkpoints"
    if not checkpoint_root.exists():
        return None
    markers = sorted(
        checkpoint_root.glob("step_*.complete.json"),
        key=lambda path: int(path.name.split("_")[1].split(".")[0]),
    )
    for marker in reversed(markers):
        state = json.loads(marker.read_text(encoding="utf-8"))
        relative_path = state.get("method_checkpoints", {}).get(method_root.name)
        if relative_path:
            path = dataset_root / relative_path
            if (path / "online_state.json").is_file():
                return path
    return None


def finalize_run(framework: ACEFramework, method_root: Path) -> None:
    framework.run_status = "completed"
    final_dir = method_root / "final"
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".final.tmp-", dir=method_root))
    try:
        framework.save_results(str(temp_dir))
        metadata_path = temp_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["config"]["experiment"]["output_dir"] = str(final_dir)
        metadata["online_protocol_version"] = PROTOCOL_VERSION
        metadata["online_source_hashes"] = current_source_hashes()
        write_json(metadata_path, metadata)
        if final_dir.exists():
            if any(final_dir.iterdir()):
                raise FileExistsError(f"Final results directory is not empty: {final_dir}")
            final_dir.rmdir()
        temp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def prepare_online_method(
    method_root: Path,
    method_name: str,
    line: str,
    args,
    dataset_key: str,
    spec: dict,
    samples: List[dict],
    method_spec: dict | None = None,
    frozen: bool = False,
) -> Tuple[ACEFramework | None, Path | None, int, dict]:
    final_dir = method_root / "final"
    method_spec = method_spec or {"line": line}
    config = make_config(
        args.model,
        spec["name"],
        ROOT / spec["stream"],
        spec["split"],
        line,
        final_dir,
        args.seed,
        len(samples),
        args=args,
        method_spec=method_spec,
        frozen=frozen,
    )
    if final_dir.exists():
        metadata_path = final_dir / "run_metadata.json"
        table_path = final_dir / "results_table.csv"
        if metadata_path.is_file() and table_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_ids = [sample["sample_id"] for sample in samples]
            if (
                metadata.get("api_model_id") == args.model
                and metadata.get("run_status") == "completed"
                and metadata.get("online_protocol_version") == PROTOCOL_VERSION
                and metadata.get("online_source_hashes") == current_source_hashes()
                and metadata.get("processed_sample_ids") == expected_ids
                and metadata.get("config") == config.to_dict()
            ):
                print(f"  {method_name}: reusing verified complete online stream", flush=True)
                return None, table_path, len(samples), {"next_index": len(samples), "sample_ids": expected_ids}
        if any(final_dir.iterdir()):
            raise FileExistsError(f"Existing final results do not match this online protocol: {final_dir}")

    previous = latest_checkpoint(method_root)
    if previous is None:
        ensure_empty(method_root)

    framework = ACEFramework(config)
    artifact_path = method_spec.get("artifact_path")
    if artifact_path:
        if not Path(artifact_path).is_file():
            raise FileNotFoundError(f"Transfer artifact not found for {method_name}: {artifact_path}")
        if method_spec.get("artifact_kind") == "skills":
            framework.skill_library.load(str(artifact_path))
        elif method_spec.get("artifact_kind") == "playbook":
            framework.playbook.load(str(artifact_path))
    state = {"next_index": 0, "sample_ids": [sample["sample_id"] for sample in samples]}
    if previous:
        runner_state = framework.load_checkpoint(str(previous))
        if runner_state.get("sample_ids") != state["sample_ids"]:
            raise ValueError(f"Checkpoint stream IDs do not match current stream: {previous}")
        if runner_state.get("source_sha256") != current_source_hashes():
            raise ValueError(f"Checkpoint code protocol differs from the current version: {previous}")
        marker_path = checkpoint_commit_path(method_root.parent, framework.metrics.processed)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("sample_ids") != state["sample_ids"][:framework.metrics.processed]:
            raise ValueError(f"Committed checkpoint sample prefix does not match: {marker_path}")
        state = runner_state
        print(
            f"  {method_name}: resumed after {framework.metrics.processed}/{len(samples)} online items",
            flush=True,
        )
    else:
        method_root.mkdir(parents=True, exist_ok=True)
        print(f"  {method_name}: starting {len(samples)}-item online stream", flush=True)

    start = framework.metrics.processed
    if start != int(state.get("next_index", -1)) or start > len(samples):
        raise ValueError(f"Invalid online checkpoint position for {method_name}: {start}")
    return framework, None, start, state


def run_online_dataset(
    dataset_key: str,
    spec: dict,
    samples: List[dict],
    args,
    methods: dict | None = None,
    frozen: bool = False,
) -> Dict[str, Path]:
    methods = methods or METHODS
    active = {}
    result_files = {}
    completed_methods = set()
    for method_name, method_spec in methods.items():
        if isinstance(method_spec, str):
            method_spec = {"line": method_spec}
        line = method_spec["line"]
        method_root = args.output_root / dataset_key / method_name
        framework, table_path, start, state = prepare_online_method(
            method_root, method_name, line, args, dataset_key, spec, samples, method_spec, frozen,
        )
        if framework is None:
            result_files[method_name] = table_path
            completed_methods.add(method_name)
        else:
            active[method_name] = {
                "framework": framework,
                "method_root": method_root,
                "next_index": start,
                "state": state,
            }

    positions = {item["next_index"] for item in active.values()}
    if len(positions) > 1:
        raise ValueError(
            f"Methods have different committed online positions in {dataset_key}: {sorted(positions)}. "
            "Do not combine partial runs."
        )
    start = next(iter(positions), len(samples))
    if completed_methods and active and start != len(samples):
        raise ValueError(
            f"Cannot mix completed and partial online method states in {dataset_key}; "
            "resume only when every remaining method is at the final shared boundary."
        )

    method_order = list(methods)
    for index in range(start, len(samples)):
        prediction_order = method_order[index % len(method_order):] + method_order[:index % len(method_order)]
        pending = {}
        predictors = [
            method_name for method_name in prediction_order
            if method_name in active and index >= active[method_name]["next_index"]
        ]

        def predict(method_name: str):
            framework = active[method_name]["framework"]
            try:
                return framework.predict_sample(samples[index], verbose=False, index=index)
            except Exception as exc:
                logging.exception("Prediction failed at %s/%s step %s", dataset_key, method_name, index + 1)
                return framework._failed_sample(samples[index], str(exc))

        parallel_size = max(1, int(getattr(args, "batch_parallel_size", 1)))
        if parallel_size == 1 or len(predictors) <= 1:
            for method_name in predictors:
                pending[method_name] = predict(method_name)
        else:
            with ThreadPoolExecutor(max_workers=min(parallel_size, len(predictors))) as executor:
                futures = {executor.submit(predict, method_name): method_name for method_name in predictors}
                for future in as_completed(futures):
                    method_name = futures[future]
                    pending[method_name] = future.result()

        for method_name in prediction_order:
            if method_name not in pending:
                continue
            item = active[method_name]
            item["framework"].observe_sample(pending[method_name], verbose=False)
            item["next_index"] = index + 1
            item["state"]["next_index"] = index + 1

        if (index + 1) % CHECKPOINT_INTERVAL == 0 or index + 1 == len(samples):
            committed_checkpoints = {}
            for method_name, item in active.items():
                if item["next_index"] != index + 1:
                    continue
                framework = item["framework"]
                framework.run_status = "running"
                checkpoint = checkpoint_path(item["method_root"], index + 1)
                item["state"]["source_sha256"] = current_source_hashes()
                if checkpoint.exists():
                    continue
                committed_checkpoint = framework.save_checkpoint(str(checkpoint), item["state"])
                committed_checkpoint = Path(committed_checkpoint or checkpoint)
                committed_checkpoints[method_name] = str(committed_checkpoint.relative_to(args.output_root / dataset_key))
                token_usage = token_usage_snapshot(framework)
                print(
                    f"  {method_name}: online step {index + 1}/{len(samples)} "
                    f"cumulative EM={framework.metrics.get_accuracy():.2f}% "
                    f"tokens={token_usage['total_tokens']:,}",
                    flush=True,
                )
            marker_path = checkpoint_commit_path(args.output_root / dataset_key, index + 1)
            if not marker_path.exists():
                if len(committed_checkpoints) != len(active):
                    raise RuntimeError(f"Incomplete method checkpoints at step {index + 1}; no commit marker written")
                write_json(marker_path, {
                    "processed": index + 1,
                    "sample_ids": [sample["sample_id"] for sample in samples[:index + 1]],
                    "method_checkpoints": committed_checkpoints,
                })

    active_positions = {item["next_index"] for item in active.values()}
    if len(active_positions) > 1:
        raise ValueError(
            f"Methods completed different numbers of online items in {dataset_key}: "
            f"{sorted(active_positions)}. Do not compare these runs."
        )

    for method_name, item in active.items():
        finalize_run(item["framework"], item["method_root"])
        result_files[method_name] = item["method_root"] / "final" / "results_table.csv"
    return result_files


def load_table(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def token_usage_snapshot(framework: ACEFramework) -> dict:
    solver_prompt = int(getattr(framework.metrics, "generator_prompt_tokens", 0) or 0)
    solver_completion = int(getattr(framework.metrics, "generator_completion_tokens", 0) or 0)
    reflection = getattr(framework, "reflection_usage", {})
    distillation = getattr(framework, "distillation_usage", {})
    reflection_prompt = int(reflection.get("prompt_tokens", 0) or 0)
    reflection_completion = int(reflection.get("completion_tokens", 0) or 0)
    distillation_prompt = int(distillation.get("prompt_tokens", 0) or 0)
    distillation_completion = int(distillation.get("completion_tokens", 0) or 0)
    return {
        "solver_prompt_tokens": solver_prompt,
        "solver_completion_tokens": solver_completion,
        "reflection_curation_prompt_tokens": reflection_prompt,
        "reflection_curation_completion_tokens": reflection_completion,
        "distillation_prompt_tokens": distillation_prompt,
        "distillation_completion_tokens": distillation_completion,
        "total_tokens": (
            solver_prompt + solver_completion
            + reflection_prompt + reflection_completion
            + distillation_prompt + distillation_completion
        ),
    }


def aggregate_online(
    dataset_key: str,
    method_name: str,
    model_name: str,
    rows: List[dict],
    metadata: dict | None = None,
) -> dict:
    count = len(rows)
    correct = sum(str(row["correct"]).lower() == "true" for row in rows)
    f1_values = [float(row["f1"]) for row in rows if row.get("f1") not in (None, "")]
    finish_reasons: Dict[str, int] = {}
    for row in rows:
        reason = row.get("finish_reason", "unknown") or "unknown"
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    parse_failures = sum(str(row.get("parse_failed", "")).lower() == "true" for row in rows)
    request_failures = sum(count for reason, count in finish_reasons.items() if reason.startswith("request_failed"))
    metadata = metadata or {}
    totals = metadata.get("measured_total_token_totals", {})
    solver = metadata.get("solver_token_totals", {})
    reflection = metadata.get("reflection_and_curation_usage", {})
    distillation = metadata.get("distillation_usage", {})
    solver_prompt_tokens = int(solver.get("prompt_tokens", 0) or 0)
    solver_completion_tokens = int(solver.get("completion_tokens", 0) or 0)
    reflection_prompt_tokens = int(reflection.get("prompt_tokens", 0) or 0)
    reflection_completion_tokens = int(reflection.get("completion_tokens", 0) or 0)
    distillation_prompt_tokens = int(distillation.get("prompt_tokens", 0) or 0)
    distillation_completion_tokens = int(distillation.get("completion_tokens", 0) or 0)
    total_prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
    total_completion_tokens = int(totals.get("completion_tokens", 0) or 0)
    total_tokens = total_prompt_tokens + total_completion_tokens
    adaptation_tokens = (
        reflection_prompt_tokens + reflection_completion_tokens
        + distillation_prompt_tokens + distillation_completion_tokens
    )
    return {
        "dataset": dataset_key,
        "method": method_name,
        "model": model_name,
        "n": count,
        "EM_percent": 100.0 * correct / count if count else 0.0,
        "F1_percent": sum(f1_values) / len(f1_values) if f1_values else None,
        "parse_failures": parse_failures,
        "request_failures": request_failures,
        "truncated_outputs": sum(str(row.get("truncated", "")).lower() == "true" for row in rows),
        "mean_solver_input_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows) / max(1, count),
        "mean_solver_output_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows) / max(1, count),
        "solver_input_tokens": solver_prompt_tokens,
        "solver_output_tokens": solver_completion_tokens,
        "solver_total_tokens": solver_prompt_tokens + solver_completion_tokens,
        "reflection_curation_input_tokens": reflection_prompt_tokens,
        "reflection_curation_output_tokens": reflection_completion_tokens,
        "reflection_curation_total_tokens": reflection_prompt_tokens + reflection_completion_tokens,
        "distillation_input_tokens": distillation_prompt_tokens,
        "distillation_output_tokens": distillation_completion_tokens,
        "distillation_total_tokens": distillation_prompt_tokens + distillation_completion_tokens,
        "total_input_tokens": total_prompt_tokens,
        "total_output_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "mean_total_input_tokens": total_prompt_tokens / max(1, count),
        "mean_total_output_tokens": total_completion_tokens / max(1, count),
        "mean_total_tokens_per_item": total_tokens / max(1, count),
        "adaptation_token_share_percent": 100.0 * adaptation_tokens / total_tokens if total_tokens else 0.0,
        "reflection_calls": int(metadata.get("reflection_calls", 0) or 0),
        "curation_calls": int(metadata.get("curation_calls", 0) or 0),
        "skill_update_calls": len(metadata.get("lifecycle_history", [])),
        "run_status": "completed_with_request_failures" if request_failures else "completed",
    }


def make_curves(result_files: Dict[str, Dict[str, Path]]) -> List[dict]:
    curves = []
    for dataset, methods in result_files.items():
        method_rows = {method: load_table(path) for method, path in methods.items()}
        if not method_rows:
            continue
        reference_method = next(iter(method_rows))
        reference_ids = [row["sample_id"] for row in method_rows[reference_method]]
        for method, rows in method_rows.items():
            ids = [row["sample_id"] for row in rows]
            if ids != reference_ids:
                raise ValueError(f"Online sample order mismatch in {dataset}/{method}")
        for start in range(0, len(reference_ids), SEGMENT_SIZE):
            end = min(start + SEGMENT_SIZE, len(reference_ids))
            for method, rows in method_rows.items():
                segment = rows[start:end]
                summary = aggregate_online(dataset, method, "", segment)
                curves.append({
                    "dataset": dataset,
                    "method": method,
                    "step_start": start + 1,
                    "step_end": end,
                    "n": summary["n"],
                    "segment_EM_percent": summary["EM_percent"],
                    "cumulative_EM_percent": aggregate_online(dataset, method, "", rows[:end])["EM_percent"],
                    "segment_F1_percent": summary["F1_percent"],
                    "cumulative_F1_percent": aggregate_online(dataset, method, "", rows[:end])["F1_percent"],
                })
    return curves


def write_outputs(
    output_root: Path,
    rows: List[dict],
    curves: List[dict],
    prefix: str = "online_main",
    title: str = "Online Prequential Main Experiment",
) -> None:
    stream_size = rows[0]["n"] if rows else 0
    columns = [
        "dataset", "method", "model", "n", "EM_percent",
        "F1_percent", "parse_failures", "request_failures", "truncated_outputs",
        "mean_solver_input_tokens", "mean_solver_output_tokens",
        "solver_input_tokens", "solver_output_tokens", "solver_total_tokens",
        "reflection_curation_input_tokens", "reflection_curation_output_tokens", "reflection_curation_total_tokens",
        "distillation_input_tokens", "distillation_output_tokens", "distillation_total_tokens",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        "mean_total_input_tokens", "mean_total_output_tokens", "mean_total_tokens_per_item",
        "adaptation_token_share_percent", "reflection_calls", "curation_calls", "skill_update_calls", "run_status",
    ]
    with (output_root / f"{prefix}_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    token_columns = [
        "dataset", "method", "model", "n",
        "solver_input_tokens", "solver_output_tokens", "solver_total_tokens",
        "reflection_curation_input_tokens", "reflection_curation_output_tokens", "reflection_curation_total_tokens",
        "distillation_input_tokens", "distillation_output_tokens", "distillation_total_tokens",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        "mean_total_tokens_per_item", "adaptation_token_share_percent",
    ]
    with (output_root / f"{prefix}_token_usage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=token_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    curve_columns = [
        "dataset", "method", "step_start", "step_end", "n", "segment_EM_percent",
        "cumulative_EM_percent", "segment_F1_percent", "cumulative_F1_percent",
    ]
    with (output_root / "online_learning_curves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_columns)
        writer.writeheader()
        writer.writerows(curves)

    lines = [
        f"# {title}",
        "",
        f"Each dataset uses one shared ordered stream of {stream_size} unique items. Every method predicts before its label is revealed; its state updates before the next item. No separate adaptation/test split is used.",
        "EM and F1 are percentages. HotpotQA reports both EM and token F1. This single-seed online score is descriptive; no independent-sample confidence interval or significance test is reported.",
        "",
        "| Dataset | Method | Model | Online items | EM (%) | F1 (%) | Parse failures | Request failures | Truncated | Total input tokens/item | Total output tokens/item | Reflection calls | Curation calls | Skill updates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f1 = "N/A" if row["F1_percent"] is None else f"{row['F1_percent']:.2f}"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['model']} | {row['n']} | "
            f"{row['EM_percent']:.2f} | "
            f"{f1} | {row['parse_failures']} | {row['request_failures']} | {row['truncated_outputs']} | "
            f"{row['mean_total_input_tokens']:.1f} | {row['mean_total_output_tokens']:.1f} | "
            f"{row['reflection_calls']} | {row['curation_calls']} | {row['skill_update_calls']} |"
        )
    lines.extend([
        "",
        "## Token Usage",
        "",
        "| Dataset | Method | Solver tokens | Reflection/curation | Distillation | Total tokens | Tokens/item | Adaptation share (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['solver_total_tokens']} | "
            f"{row['reflection_curation_total_tokens']} | {row['distillation_total_tokens']} | "
            f"{row['total_tokens']} | {row['mean_total_tokens_per_item']:.1f} | "
            f"{row['adaptation_token_share_percent']:.2f} |"
        )
    lines.extend([
        "",
        "## Comparison Protocol",
        "",
        "Methods are paired by identical item IDs and order. Since online predictions depend on each method's preceding feedback and evolving state, ordinary independent-sample tests are not reported. Use the saved per-item outcomes and learning curves for paired descriptive analysis; repeated seeds are needed for inferential claims.",
        "",
        "The baseline receives no feedback-driven memory updates. ACE updates its playbook after each answer through reflection and curation. MUSE updates skill feedback after each answer and distills recent trajectories at the configured interval; both adaptive methods see only labels from already-predicted items. End-to-end token costs include solver, reflection/curation, and skill-distillation calls.",
    ])
    (output_root / f"{prefix}_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_frozen_results(output_root: Path, rows: List[dict], title: str) -> None:
    columns = list(rows[0]) if rows else []
    with (output_root / "frozen_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    token_columns = [
        "dataset", "method", "model", "n",
        "solver_input_tokens", "solver_output_tokens", "solver_total_tokens",
        "reflection_curation_input_tokens", "reflection_curation_output_tokens", "reflection_curation_total_tokens",
        "distillation_input_tokens", "distillation_output_tokens", "distillation_total_tokens",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        "mean_total_tokens_per_item", "adaptation_token_share_percent",
    ]
    with (output_root / "frozen_token_usage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=token_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {title}",
        "",
        "All methods are evaluated on the same frozen holdout IDs. No holdout feedback is used for updates.",
        "",
        "| Dataset | Method | Model | N | EM (%) | F1 (%) | Total tokens | Tokens/item | Parse failures | Request failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f1 = "N/A" if row["F1_percent"] is None else f"{row['F1_percent']:.2f}"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['model']} | {row['n']} | "
            f"{row['EM_percent']:.2f} | {f1} | {row['total_tokens']} | "
            f"{row['mean_total_tokens_per_item']:.1f} | {row['parse_failures']} | {row['request_failures']} |"
        )
    (output_root / "frozen_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ablation_methods() -> dict:
    return {
        "base_llm": {"line": "base_llm"},
        "ACE": {"line": "baseline"},
        "MUSE": {"line": "skillline"},
        "MUSE_single_trajectory": {
            "line": "skillline",
            "overrides": {"trajectory_window_size": 1},
        },
        "MUSE_no_gate": {
            "line": "skillline",
            "overrides": {"use_uncertainty_gate": False},
        },
        "MUSE_no_lifecycle": {
            "line": "skillline",
            "overrides": {"use_lifecycle_management": False},
        },
    }


def transfer_methods(source_run: Path, source_dataset: str) -> dict:
    source_dir = source_run / "online" / source_dataset
    return {
        "base_llm": {"line": "base_llm"},
        "MUSE_transfer": {
            "line": "skillline",
            "artifact_kind": "skills",
            "artifact_path": str(source_dir / "MUSE" / "final" / "ace_skills.json"),
            "frozen": True,
        },
    }


def experiment_parameters(args) -> dict:
    return {
        "model": args.model,
        "seed": args.seed,
        "stream_size": args.stream_size,
        "holdout_size": args.holdout_size,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "batch_parallel_size": args.batch_parallel_size,
        "enable_thinking": args.enable_thinking,
        "skill_update_interval": args.skill_update_interval,
        "trajectory_window_size": args.trajectory_window_size,
        "retrieval_pool_size": args.retrieval_pool_size,
        "skill_top_k": args.skill_top_k,
        "gate_alpha": args.gate_alpha,
        "gate_beta": args.gate_beta,
        "gate_gamma": args.gate_gamma,
        "gate_delta": args.gate_delta,
        "gate_threshold": args.gate_threshold,
        "gate_uncertainty_adjustment": args.gate_uncertainty_adjustment,
        "gate_helpfulness_adjustment": args.gate_helpfulness_adjustment,
        "retain_threshold": args.retain_threshold,
        "retire_threshold": args.retire_threshold,
        "max_library_size": args.max_library_size,
        "utility_confidence_weight": args.utility_confidence_weight,
        "utility_helpfulness_weight": args.utility_helpfulness_weight,
        "utility_usage_weight": args.utility_usage_weight,
        "utility_redundancy_weight": args.utility_redundancy_weight,
    }


def aggregate_results(result_files: dict, model: str) -> list[dict]:
    rows = []
    for dataset_key, methods in result_files.items():
        for method_name, table_path in methods.items():
            table_rows = load_table(table_path)
            metadata_path = table_path.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append(aggregate_online(dataset_key, method_name, model, table_rows, metadata))
    return rows


def run_experiment(args) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"revision_{args.experiment}_{args.model}_{args.seed}_{stamp}"
    output_root = (args.output_root or ROOT / "results" / default_name).resolve()
    manifest_path = output_root / "experiment_manifest.json"
    if output_root.exists() and any(output_root.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Refusing to use a non-empty directory without an experiment manifest: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    args.output_root = output_root / "online"
    args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = experiment_parameters(args)
    source_hashes = current_source_hashes()

    if args.experiment == "transfer":
        if not args.source_run or not args.source_dataset or not args.target_dataset:
            raise ValueError("transfer requires --source-run, --source-dataset, and --target-dataset")
        source_run = args.source_run.expanduser().resolve()
        source_manifest_path = source_run / "experiment_manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"Source main-experiment manifest not found: {source_manifest_path}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if (
            source_manifest.get("experiment") != "main"
            or source_manifest.get("protocol_version") != PROTOCOL_VERSION
            or not source_manifest.get("completed_at")
        ):
            raise ValueError("Transfer source must be a completed main experiment")
        if source_manifest.get("model") != args.model:
            raise ValueError("Transfer and source runs must use the same model")
        source_dataset = args.source_dataset
        target_dataset = args.target_dataset
        if source_dataset == target_dataset:
            raise ValueError("Cross-dataset transfer requires different --source-dataset and --target-dataset")
        if source_manifest.get("seed") != args.seed:
            raise ValueError("Transfer must use the same --seed as the source main run")
        target_spec = DATASETS[target_dataset]
        target_info = source_manifest.get("datasets", {}).get(target_dataset, {}).get("holdout")
        if not target_info:
            raise ValueError(f"Source run has no frozen holdout manifest for {target_dataset}")
        methods = transfer_methods(source_run, source_dataset)
        for name, spec in methods.items():
            if name != "base_llm" and not Path(spec["artifact_path"]).is_file():
                raise FileNotFoundError(f"Missing source artifact: {spec['artifact_path']}")
        samples = samples_from_manifest(target_spec, target_info)
        manifest = {
            "experiment": "transfer",
            "protocol_version": PROTOCOL_VERSION,
            "model": args.model,
            "seed": args.seed,
            "source_run": str(source_run),
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "source_artifacts": {
                name: {"path": spec["artifact_path"], "sha256": sha256(Path(spec["artifact_path"]))}
                for name, spec in methods.items() if name != "base_llm"
            },
            "holdout": target_info,
            "parameters": parameters,
            "source_sha256": source_hashes,
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(previous.get(key) != manifest.get(key) for key in manifest):
                raise ValueError("Output directory belongs to a different transfer configuration")
            manifest = previous
        else:
            write_json(manifest_path, manifest)
        print(f"[{source_dataset} -> {target_dataset}] frozen transfer evaluation n={len(samples)}", flush=True)
        result_files = run_online_dataset(
            f"{source_dataset}_to_{target_dataset}", target_spec, samples, args, methods, frozen=True,
        )
        rows = aggregate_results({target_dataset: result_files}, args.model)
        write_frozen_results(output_root, rows, f"Frozen Transfer: {source_dataset} to {target_dataset}")
        manifest["completed_at"] = datetime.now().isoformat()
        write_json(manifest_path, manifest)
        return output_root

    selected_datasets = {key: DATASETS[key] for key in args.datasets}

    if args.experiment == "main":
        selected_method_names = getattr(args, "methods", None) or list(METHODS)
        methods = {name: {"line": METHODS[name]} for name in selected_method_names}
    else:
        methods = ablation_methods()
    splits = {}
    for dataset_key, spec in selected_datasets.items():
        train, train_info, holdout, holdout_info = choose_splits(
            spec,
            args.seed,
            args.stream_size,
            args.holdout_size if args.experiment == "main" else 0,
        )
        splits[dataset_key] = (train, train_info, holdout, holdout_info)

    manifest = {
        "experiment": args.experiment,
        "protocol_version": PROTOCOL_VERSION,
        "model": args.model,
        "seed": args.seed,
        "methods": methods,
        "parameters": parameters,
        "source_sha256": source_hashes,
        "protocol": (
            "Matched prequential online stream; all methods predict before feedback. "
            "A disjoint frozen holdout is evaluated only when holdout_size is greater than zero."
        ),
        "datasets": {
            key: {"online": train_info, **({"holdout": holdout_info} if holdout_info else {})}
            for key, (_, train_info, _, holdout_info) in splits.items()
        },
        "started_at": datetime.now().isoformat(),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("completed_at"):
            raise FileExistsError(f"This experiment is already finalized: {output_root}")
        for key in ("experiment", "protocol_version", "model", "seed", "methods", "parameters", "source_sha256", "datasets"):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Existing output root does not match current experiment field: {key}")
        manifest = previous
    else:
        write_json(manifest_path, manifest)

    online_root = args.output_root
    online_files = {}
    online_rows = []
    for dataset_key, spec in selected_datasets.items():
        train, _, _, _ = splits[dataset_key]
        print(f"\n[{dataset_key}] shared online stream n={len(train)}; seed={args.seed}", flush=True)
        online_files[dataset_key] = run_online_dataset(dataset_key, spec, train, args, methods)
    online_rows = aggregate_results(online_files, args.model)
    curves = make_curves(online_files)
    write_outputs(
        online_root,
        online_rows,
        curves,
        prefix=f"online_{args.experiment}",
        title=f"Online Prequential {args.experiment.title()} Experiment",
    )

    if args.experiment == "main" and args.holdout_size > 0:
        holdout_root = output_root / "holdout"
        holdout_root.mkdir(parents=True, exist_ok=True)
        holdout_methods = {}
        for key in selected_datasets:
            holdout_methods[key] = {}
            if "base_llm" in methods:
                holdout_methods[key]["base_llm"] = {"line": "base_llm"}
            if "ACE" in methods:
                holdout_methods[key]["ACE"] = {
                    "line": "baseline", "artifact_kind": "playbook", "frozen": True,
                    "artifact_path": str(output_root / "online" / key / "ACE" / "final" / "ace_playbook.json"),
                }
            if "MUSE" in methods:
                holdout_methods[key]["MUSE"] = {
                    "line": "skillline", "artifact_kind": "skills", "frozen": True,
                    "artifact_path": str(output_root / "online" / key / "MUSE" / "final" / "ace_skills.json"),
                }
        holdout_files = {}
        args.output_root = holdout_root
        for dataset_key, spec in selected_datasets.items():
            _, _, holdout, _ = splits[dataset_key]
            print(f"[{dataset_key}] frozen holdout n={len(holdout)}", flush=True)
            holdout_files[dataset_key] = run_online_dataset(
                dataset_key, spec, holdout, args, holdout_methods[dataset_key], frozen=True,
            )
        frozen_rows = aggregate_results(holdout_files, args.model)
        write_frozen_results(holdout_root, frozen_rows, "Frozen Holdout Evaluation")
        manifest["frozen_holdout_results"] = str(holdout_root / "frozen_results.md")

    manifest["online_results"] = str(online_root / f"online_{args.experiment}_results.md")
    manifest["completed_at"] = datetime.now().isoformat()
    manifest["output_root"] = str(output_root)
    write_json(manifest_path, manifest)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("main", "ablation", "transfer"), default="main")
    parser.add_argument("--model", default="qwen3.5-35b-a3b")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Datasets to run sequentially.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(METHODS),
        help="Main-experiment methods to run on the shared online stream.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--stream-size", type=int, default=STREAM_SIZE)
    parser.add_argument("--holdout-size", type=int, default=HOLDOUT_SIZE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-dataset", choices=tuple(DATASETS))
    parser.add_argument("--target-dataset", choices=tuple(DATASETS))
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--batch-parallel-size", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--skill-update-interval", type=int, default=100)
    parser.add_argument("--trajectory-window-size", type=int, default=20)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--skill-top-k", type=int, default=8)
    parser.add_argument("--gate-alpha", type=float, default=0.45)
    parser.add_argument("--gate-beta", type=float, default=0.25)
    parser.add_argument("--gate-gamma", type=float, default=0.20)
    parser.add_argument("--gate-delta", type=float, default=0.10)
    parser.add_argument("--gate-threshold", type=float, default=0.45)
    parser.add_argument("--gate-uncertainty-adjustment", type=float, default=0.10)
    parser.add_argument("--gate-helpfulness-adjustment", type=float, default=0.05)
    parser.add_argument("--retain-threshold", type=float, default=0.20)
    parser.add_argument("--retire-threshold", type=float, default=0.05)
    parser.add_argument("--max-library-size", type=int, default=32)
    parser.add_argument("--utility-confidence-weight", type=float, default=0.50)
    parser.add_argument("--utility-helpfulness-weight", type=float, default=0.25)
    parser.add_argument("--utility-usage-weight", type=float, default=0.05)
    parser.add_argument("--utility-redundancy-weight", type=float, default=0.20)
    args = parser.parse_args()
    if args.stream_size < 1 or args.holdout_size < 0:
        parser.error("--stream-size must be >= 1 and --holdout-size must be >= 0")
    if args.retrieval_pool_size < args.skill_top_k:
        parser.error("--retrieval-pool-size must be >= --skill-top-k")
    if args.batch_parallel_size < 1:
        parser.error("--batch-parallel-size must be >= 1")
    if args.experiment == "transfer" and not all((args.source_run, args.source_dataset, args.target_dataset)):
        parser.error("transfer requires --source-run, --source-dataset, and --target-dataset")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    output_root = run_experiment(args)
    print(f"\nExperiment outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
