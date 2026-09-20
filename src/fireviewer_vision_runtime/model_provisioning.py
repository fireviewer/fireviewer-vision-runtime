"""Provision pinned public model weights into a mounted persistent volume."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fireviewer_contracts.model_registry import ModelSpec, enabled_public_models
from fireviewer_geolocation.roma_registration import ROMA_ASSETS, provision_roma_assets

MANIFEST_NAME = "firewarning-model-cache.json"


@dataclass(frozen=True, slots=True)
class CacheStatus:
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


def _selected_models(*, skip_qwen: bool) -> tuple[ModelSpec, ...]:
    return tuple(
        spec
        for spec in enabled_public_models()
        if not (skip_qwen and spec.role == "multimodal_extraction")
    )


def _manifest_path(cache_root: Path) -> Path:
    return cache_root.resolve().parent / MANIFEST_NAME


def _snapshot_path(cache_root: Path, spec: ModelSpec) -> Path:
    repository = cache_root / f"models--{spec.model_id.replace('/', '--')}" / "snapshots"
    return repository / spec.revision


def _snapshot_weight_files(snapshot: Path) -> tuple[Path, ...]:
    if not snapshot.is_dir():
        return ()
    if "models--prism-ml--Ternary-Bonsai-2-27B-gguf" in snapshot.parts:
        expected = {'Ternary-Bonsai-2-27B-PTQ1_0.gguf': 5946648928, 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf': 629246976}
        return tuple(snapshot / name for name in expected) if all(
            (snapshot / name).is_file() and (snapshot / name).stat().st_size == size
            for name, size in expected.items()) else ()
    weight_suffixes = {".safetensors", ".pt", ".pth", ".gguf"}
    return tuple(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.suffix.casefold() in weight_suffixes
    )


def _snapshot_size_bytes(snapshot: Path) -> int:
    return sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())


def _read_manifest(cache_root: Path) -> dict[str, Any] | None:
    path = _manifest_path(cache_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def cache_status(cache_root: Path, roma_root: Path, *, skip_qwen: bool = False) -> CacheStatus:
    """Perform a cheap cold-start check against the atomic provisioning marker."""
    cache_root = cache_root.resolve()
    roma_root = roma_root.resolve()
    expected_models = _selected_models(skip_qwen=skip_qwen)
    manifest = _read_manifest(cache_root)
    manifest_models = manifest.get("models") if manifest else None
    expected_manifest_models = [
        {"model_id": spec.model_id, "revision": spec.revision, "role": spec.role}
        for spec in expected_models
    ]
    missing: list[str] = []
    if manifest_models != expected_manifest_models:
        missing.append("provisioning_manifest")
    for spec in expected_models:
        snapshot = _snapshot_path(cache_root, spec)
        if not _snapshot_weight_files(snapshot):
            missing.append(spec.role)
    for asset_spec in ROMA_ASSETS:
        asset = roma_root / "weights" / asset_spec.filename
        if not asset.is_file() or asset.stat().st_size != asset_spec.size:
            missing.append(f"spatial_registration:{asset_spec.filename}")
    return CacheStatus(missing=tuple(dict.fromkeys(missing)))


def provision_model_cache(
    cache_root: Path,
    roma_root: Path,
    *,
    skip_qwen: bool = False,
) -> dict[str, Any]:
    """Download only missing pinned assets and publish an atomic completion marker."""
    cache_root = cache_root.resolve()
    roma_root = roma_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    roma_root.mkdir(parents=True, exist_ok=True)

    # Import only after the bootstrap explicitly enables network access. This
    # prevents Hugging Face's module-level offline state from leaking into the
    # final worker process, which is started with a fresh exec in offline mode.
    from huggingface_hub import snapshot_download

    selected_models = _selected_models(skip_qwen=skip_qwen)
    existing_manifest = _read_manifest(cache_root) or {}
    existing_entries = existing_manifest.get("models")
    marked_entries = existing_entries if isinstance(existing_entries, list) else []
    model_entries: list[dict[str, str]] = []
    for spec in selected_models:
        snapshot = _snapshot_path(cache_root, spec)
        entry = {"model_id": spec.model_id, "revision": spec.revision, "role": spec.role}
        if entry not in marked_entries or not _snapshot_weight_files(snapshot):
            print(
                "firewarning bootstrap: downloading model "
                f"role={spec.role} model={spec.model_id} revision={spec.revision}",
                flush=True,
            )
            snapshot_download(
                repo_id=spec.model_id,
                revision=spec.revision,
                cache_dir=cache_root,
                local_files_only=False,
                allow_patterns=(
                    ["Ternary-Bonsai-2-27B-PTQ1_0.gguf", "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"]
                    if spec.model_id == "prism-ml/Ternary-Bonsai-2-27B-gguf" else None
                ),
                ignore_patterns=[
                    "*.bin",
                    "*.onnx",
                    "*.msgpack",
                    "*.h5",
                    "*fp32*.safetensors",
                ],
            )
        else:
            print(
                f"firewarning bootstrap: model cache hit role={spec.role} model={spec.model_id}",
                flush=True,
            )
        if not _snapshot_weight_files(snapshot):
            raise RuntimeError(f"pinned snapshot contains no supported weights: {snapshot}")
        print(
            "firewarning bootstrap: model ready "
            f"role={spec.role} model={spec.model_id} bytes={_snapshot_size_bytes(snapshot)}",
            flush=True,
        )
        model_entries.append(entry)

    roma_manifest = provision_roma_assets(roma_root)
    manifest: dict[str, Any] = {
        "models": model_entries,
        "roma_source_revision": roma_manifest["source_revision"],
        "schema_version": 1,
        "storage_policy": "mounted_volume_no_docker_image_no_git",
    }
    manifest_path = _manifest_path(cache_root)
    partial = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    partial.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, manifest_path)
    return manifest
