from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import Sha256HexV2, StrictModel

GroundingDinoFilename = Literal[
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]
_REQUIRED_FILES: tuple[GroundingDinoFilename, ...] = (
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


class LocalGroundingDinoFile(StrictModel):
    filename: GroundingDinoFilename
    byte_size: int = Field(gt=0)
    sha256: Sha256HexV2


class LocalGroundingDinoBundleManifest(StrictModel):
    schema_name: Literal["fireviewer.local-grounding-dino-bundle.v1"] = Field(
        default="fireviewer.local-grounding-dino-bundle.v1",
        alias="schema",
        serialization_alias="schema",
    )
    model_id: Literal["IDEA-Research/grounding-dino-tiny"] = "IDEA-Research/grounding-dino-tiny"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: tuple[LocalGroundingDinoFile, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_files(self) -> LocalGroundingDinoBundleManifest:
        filenames = [item.filename for item in self.files]
        if set(filenames) != set(_REQUIRED_FILES) or len(filenames) != len(set(filenames)):
            raise ValueError("local Grounding DINO bundle must contain its required files")
        return self


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_local_grounding_dino_bundle(
    directory: Path,
    *,
    revision: str,
) -> LocalGroundingDinoBundleManifest:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Grounding DINO revision must be an immutable 40-character commit")
    root = directory.resolve(strict=True)
    files: list[LocalGroundingDinoFile] = []
    for filename in _REQUIRED_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"local Grounding DINO bundle is missing {filename}")
        metadata_path = root / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
        if metadata_path.is_file():
            metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
            if not metadata_lines or metadata_lines[0] != revision:
                raise ValueError("Grounding DINO Hugging Face receipt revision does not match")
        files.append(
            LocalGroundingDinoFile(
                filename=filename,
                byte_size=path.stat().st_size,
                sha256=_file_digest(path),
            )
        )
    manifest = LocalGroundingDinoBundleManifest(revision=revision, files=tuple(files))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("architectures") != [
        "GroundingDinoForObjectDetection"
    ]:
        raise ValueError("local Grounding DINO config declares an unexpected architecture")
    if config.get("model_type") != "grounding-dino":
        raise ValueError("local Grounding DINO config declares an unexpected model type")
    return manifest


class LocalGroundingDinoModelLoader:
    """Load a digest-qualified Grounding DINO model without network access."""

    def __init__(
        self,
        *,
        directory: Path,
        manifest: LocalGroundingDinoBundleManifest,
    ) -> None:
        self.directory = directory.resolve(strict=True)
        self.manifest = manifest

    def __call__(self) -> tuple[Any, Any]:
        for expected in self.manifest.files:
            path = self.directory / expected.filename
            if (
                not path.is_file()
                or path.stat().st_size != expected.byte_size
                or _file_digest(path) != expected.sha256
            ):
                raise ValueError("local Grounding DINO bundle no longer matches its manifest")

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor_loader: Any = AutoProcessor.from_pretrained
        processor = processor_loader(
            self.directory,
            local_files_only=True,
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.directory,
            local_files_only=True,
            use_safetensors=True,
        )
        model.eval()
        return processor, model


__all__ = [
    "LocalGroundingDinoBundleManifest",
    "LocalGroundingDinoFile",
    "LocalGroundingDinoModelLoader",
    "inspect_local_grounding_dino_bundle",
]
