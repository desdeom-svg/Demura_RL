import argparse
import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional


DEFAULT_ARCHIVE = r"E:\Projects\DemuraAI_RL\Demo3\RealWorld_Train\optimization_history.jsonl"
DEFAULT_HASH_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_SAMPLE_HASH_BYTES = 1024 * 1024


def load_archive(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_failure_summary(save_dir: str) -> Optional[Dict[str, Any]]:
    summary_path = os.path.join(save_dir, "failure_artifacts", "reset_failure_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_artifact_metadata(save_dir: str, copied_inputs: List[str]) -> Dict[str, Dict[str, Any]]:
    artifacts_dir = os.path.join(save_dir, "failure_artifacts")
    metadata: Dict[str, Dict[str, Any]] = {}
    candidate_names = list(copied_inputs)
    panel_name = "reset_failure_panel.bmp"
    if panel_name not in candidate_names:
        candidate_names.append(panel_name)

    for name in candidate_names:
        path = os.path.join(artifacts_dir, name)
        if not os.path.exists(path):
            continue
        stat = os.stat(path)
        item = {
            "size_bytes": int(stat.st_size),
            "mtime": stat.st_mtime,
        }
        if int(stat.st_size) <= DEFAULT_HASH_MAX_BYTES:
            item["sha1"] = sha1_file(path)
        else:
            item["sampled_sha1"] = sampled_sha1_file(path)
        metadata[name] = item
    return metadata


def sha1_file(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sampled_sha1_file(path: str, sample_bytes: int = DEFAULT_SAMPLE_HASH_BYTES) -> str:
    file_size = os.path.getsize(path)
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        head = handle.read(sample_bytes)
        digest.update(head)
        if file_size > sample_bytes:
            handle.seek(max(0, file_size - sample_bytes))
            tail = handle.read(sample_bytes)
            digest.update(tail)
    return digest.hexdigest()


def summarize_locator_failures(rows: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    locator_rows = [
        row for row in rows
        if row.get("failure_stage") == "locator" or row.get("status") == "hardware_locator_failure"
    ]
    locator_rows = locator_rows[-limit:]

    enriched: List[Dict[str, Any]] = []
    copied_input_signatures: Counter[str] = Counter()
    panel_signatures: Counter[str] = Counter()
    complete_evidence_input_signatures: Counter[str] = Counter()
    complete_evidence_panel_signatures: Counter[str] = Counter()
    complete_evidence_count = 0
    latest_complete_evidence: Optional[Dict[str, Any]] = None

    for row in locator_rows:
        save_dir = row.get("save_dir", "")
        summary = load_failure_summary(save_dir) if save_dir else None
        copied_inputs = (summary or {}).get("copied_inputs", [])
        copied_metadata = dict((summary or {}).get("copied_file_metadata", {}))
        filesystem_metadata = collect_artifact_metadata(save_dir, copied_inputs) if save_dir else {}
        for name, meta in filesystem_metadata.items():
            copied_metadata.setdefault(name, {})
            copied_metadata[name].setdefault("size_bytes", meta.get("size_bytes"))
            copied_metadata[name].setdefault("mtime", meta.get("mtime"))
            copied_metadata[name].setdefault("sha1", meta.get("sha1"))
            copied_metadata[name].setdefault("sampled_sha1", meta.get("sampled_sha1"))
        panel_meta = copied_metadata.get("reset_failure_panel.bmp", {})

        copied_signature = "|".join(
            (
                f"{name}:{copied_metadata.get(name, {}).get('size_bytes')}:"
                f"{copied_metadata.get(name, {}).get('sha1') or copied_metadata.get(name, {}).get('sampled_sha1')}"
            )
            for name in sorted(copied_inputs)
        )
        panel_signature = (
            f"{panel_meta.get('size_bytes')}:{panel_meta.get('sha1')}:{panel_meta.get('mtime')}"
        )

        copied_input_signatures[copied_signature] += 1
        panel_signatures[panel_signature] += 1
        complete_evidence = bool(copied_inputs) and bool(panel_meta)
        if complete_evidence:
            complete_evidence_count += 1
            complete_evidence_input_signatures[copied_signature] += 1
            complete_evidence_panel_signatures[panel_signature] += 1
            latest_complete_evidence = {
                "timestamp": row.get("timestamp"),
                "save_dir": save_dir,
                "panel_sha1": panel_meta.get("sha1"),
                "mim_fingerprints": {
                    name: (
                        copied_metadata.get(name, {}).get("sha1")
                        or copied_metadata.get(name, {}).get("sampled_sha1")
                    )
                    for name in sorted(copied_inputs)
                },
            }

        enriched.append(
            {
                "timestamp": row.get("timestamp"),
                "save_dir": save_dir,
                "status": row.get("status"),
                "failure_stage": row.get("failure_stage"),
                "failure_reason": row.get("failure_reason"),
                "copied_inputs": copied_inputs,
                "copied_file_metadata": copied_metadata,
                "render_stats": (summary or {}).get("render_stats"),
                "complete_evidence": complete_evidence,
            }
        )

    all_complete_evidence_inputs_match = len(complete_evidence_input_signatures) <= 1
    all_complete_evidence_panels_match = len(complete_evidence_panel_signatures) <= 1
    if complete_evidence_count == 0:
        consistency_verdict = "insufficient_complete_evidence"
        recommended_next_action = "capture at least one more complete-evidence locator failure or a successful locator sample"
    elif all_complete_evidence_inputs_match and all_complete_evidence_panels_match:
        consistency_verdict = "stable_inputs_stable_failures"
        recommended_next_action = "compare these stable locator-failure artifacts against a successful locator sample or template"
    else:
        consistency_verdict = "input_or_capture_variation_detected"
        recommended_next_action = "normalize acquisition inputs before drawing conclusions about locator logic"

    reference_fingerprint = None
    if latest_complete_evidence is not None:
        reference_fingerprint = {
            "source_save_dir": latest_complete_evidence.get("save_dir"),
            "source_timestamp": latest_complete_evidence.get("timestamp"),
            "panel_sha1": latest_complete_evidence.get("panel_sha1"),
            "mim_fingerprints": latest_complete_evidence.get("mim_fingerprints"),
        }

    return {
        "locator_failure_count": len(locator_rows),
        "complete_evidence_count": complete_evidence_count,
        "locator_failures": enriched,
        "copied_input_signature_counts": dict(copied_input_signatures),
        "panel_signature_counts": dict(panel_signatures),
        "all_recent_locator_inputs_match": len(copied_input_signatures) <= 1,
        "all_recent_locator_panels_match": len(panel_signatures) <= 1,
        "complete_evidence_input_signature_counts": dict(complete_evidence_input_signatures),
        "complete_evidence_panel_signature_counts": dict(complete_evidence_panel_signatures),
        "all_complete_evidence_inputs_match": all_complete_evidence_inputs_match,
        "all_complete_evidence_panels_match": all_complete_evidence_panels_match,
        "consistency_verdict": consistency_verdict,
        "recommended_next_action": recommended_next_action,
        "latest_complete_evidence": latest_complete_evidence,
        "reference_fingerprint": reference_fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recent hardware locator failure artifacts.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--write-reference-json", default="")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()

    rows = load_archive(args.archive)
    summary = summarize_locator_failures(rows, args.limit)
    reference = summary.get("reference_fingerprint")
    if args.write_reference_json:
        output_dir = os.path.dirname(args.write_reference_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.write_reference_json, "w", encoding="utf-8") as handle:
            json.dump(reference, handle, ensure_ascii=False, indent=2)
    if args.reference_only:
        print(json.dumps(reference, ensure_ascii=False, indent=2))
        return
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
