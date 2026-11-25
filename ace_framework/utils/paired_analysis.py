"""Paired analysis for per-sample experiment result CSV files."""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_results(path: Path) -> Dict[str, bool]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "correct" not in rows.fieldnames:
            raise ValueError(f"{path} has no 'correct' column")
        key_field = "sample_id" if "sample_id" in rows.fieldnames else "source_index"
        results = {}
        for row in rows:
            key = (row.get(key_field) or "").strip()
            if not key:
                raise ValueError(f"{path} contains a row without {key_field}")
            if key in results:
                raise ValueError(f"{path} contains duplicate {key_field}: {key}")
            value = (row.get("correct") or "").strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"{path} has invalid correctness value for {key}: {value!r}")
            results[key] = value in {"true", "1"}
        return results


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar_p(discordant_left: int, discordant_right: int) -> float:
    n = discordant_left + discordant_right
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(discordant_left, discordant_right) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _bootstrap_difference_interval(
    differences: List[float],
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    if not differences:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    low_index = int(0.025 * (iterations - 1))
    high_index = int(0.975 * (iterations - 1))
    return means[low_index], means[high_index]


def _load_f1_results(path: Path) -> Dict[str, float] | None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "f1" not in rows.fieldnames:
            return None
        key_field = "sample_id" if "sample_id" in rows.fieldnames else "source_index"
        scores = {}
        missing = False
        for row in rows:
            key = (row.get(key_field) or "").strip()
            value = (row.get("f1") or "").strip()
            if not value:
                missing = True
                continue
            score = float(value)
            if not 0.0 <= score <= 100.0:
                raise ValueError(f"{path} has out-of-range percentage F1 for {key}: {score}")
            scores[key] = score
        if not scores and missing:
            return None
        if missing:
            raise ValueError(f"{path} has missing F1 values in a partially scored file")
        return scores


def compare_result_files(
    left_path: Path,
    right_path: Path,
    bootstrap_iterations: int = 10000,
    seed: int = 38,
) -> dict:
    left = _load_results(left_path)
    right = _load_results(right_path)
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())[:10]
        missing_right = sorted(left.keys() - right.keys())[:10]
        raise ValueError(
            "Result files must contain the same sample IDs; "
            f"missing from first={missing_left}, missing from second={missing_right}"
        )
    keys = sorted(left)
    left_correct = sum(left[key] for key in keys)
    right_correct = sum(right[key] for key in keys)
    left_only = sum(left[key] and not right[key] for key in keys)
    right_only = sum(right[key] and not left[key] for key in keys)
    differences = [int(right[key]) - int(left[key]) for key in keys]
    left_ci = wilson_interval(left_correct, len(keys))
    right_ci = wilson_interval(right_correct, len(keys))
    delta_ci = _bootstrap_difference_interval(differences, bootstrap_iterations, seed)
    report = {
        "n": len(keys),
        "first_accuracy": left_correct / len(keys) if keys else 0.0,
        "first_accuracy_wilson_95": list(left_ci),
        "second_accuracy": right_correct / len(keys) if keys else 0.0,
        "second_accuracy_wilson_95": list(right_ci),
        "paired_accuracy_difference_second_minus_first": sum(differences) / len(keys) if keys else 0.0,
        "paired_difference_bootstrap_95": list(delta_ci),
        "first_only_correct": left_only,
        "second_only_correct": right_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(left_only, right_only),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": seed,
        "first_file": str(left_path),
        "second_file": str(right_path),
    }
    left_f1 = _load_f1_results(left_path)
    right_f1 = _load_f1_results(right_path)
    if (left_f1 is None) != (right_f1 is None):
        raise ValueError("Both result files must contain HotpotQA F1 values, or neither may")
    if left_f1 is not None and right_f1 is not None:
        if left_f1.keys() != right_f1.keys() or left_f1.keys() != left.keys():
            raise ValueError("F1 values must match the same sample IDs as exact-match results")
        f1_differences = [right_f1[key] - left_f1[key] for key in keys]
        report.update({
            "first_mean_f1": sum(left_f1.values()) / len(keys) if keys else 0.0,
            "second_mean_f1": sum(right_f1.values()) / len(keys) if keys else 0.0,
            "paired_f1_difference_second_minus_first": (
                sum(f1_differences) / len(keys) if keys else 0.0
            ),
            "paired_f1_difference_bootstrap_95": list(
                _bootstrap_difference_interval(f1_differences, bootstrap_iterations, seed)
            ),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_csv", type=Path)
    parser.add_argument("second_csv", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be >= 1")
    result = compare_result_files(
        args.first_csv, args.second_csv, args.bootstrap_iterations, args.seed
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
