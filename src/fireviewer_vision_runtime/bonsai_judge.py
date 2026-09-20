"""Pinned Bonsai 2 consensus judge for the isolated FireViewer laboratory.

Owns a loopback llama-server process, loaded after the other model candidates
have released GPU memory. Jev remains a separate, text-only experiment.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from fireviewer_contracts.model_registry import resolve_cached_snapshot
from fireviewer_vision_runtime.adapters import ModelOutputError
from fireviewer_vision_runtime.consensus import ConsensusJudgeVerdict

MODEL_ID = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
MODEL_FILE = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
VISION_FILE = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"


class BonsaiConsensusJudgeAdapter:
    def __init__(self, spec, *, cache_root, fetcher):
        if spec.model_id != MODEL_ID or spec.revision != MODEL_REVISION or spec.role != "consensus_judge":
            raise ValueError("bonsai_judge_requires_pinned_model_and_role")
        self.spec, self.cache_root, self.fetcher = spec, cache_root, fetcher
        self.process = None
        self.url = None
        self.last_receipt = None
        self.gpu_samples = []
        self.sampling_stop = threading.Event()
        self.sampling_thread = None
        self.http = build_opener(ProxyHandler({}))

    def _sample_gpu(self):
        while not self.sampling_stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--id=0',
                    '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2, check=True)
                name, total, used = [s.strip() for s in result.stdout.strip().split(',')]
                self.gpu_samples.append({'device': name, 'total_mib': int(total), 'used_mib': int(used)})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.sampling_stop.wait(1)

    def _gpu_receipt(self):
        if not self.gpu_samples:
            return {'status': 'UNMEASURED'}
        return {'status': 'SAMPLED', 'scope': 'device_0_total_usage_not_process_only',
                'interval_seconds': 1, 'samples': len(self.gpu_samples),
                'device': self.gpu_samples[0]['device'], 'total_mib': self.gpu_samples[0]['total_mib'],
                'first_used_mib': self.gpu_samples[0]['used_mib'],
                'peak_sampled_used_mib': max(s['used_mib'] for s in self.gpu_samples)}

    def load(self):
        if self.process is not None:
            raise RuntimeError("bonsai_judge_already_loaded")
        layers = os.environ.get("FW_BONSAI_GPU_LAYERS", "99")
        if layers == "0":
            if os.environ.get("FW_BONSAI_CPU_VALIDATION") != "1":
                raise RuntimeError("cpu_mode_only_for_explicit_validation")
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("bonsai_gpu_required_no_silent_cpu_fallback")
        path = resolve_cached_snapshot(self.spec, self.cache_root)
        for name in (MODEL_FILE, VISION_FILE):
            if not (path / name).is_file():
                raise RuntimeError("bonsai_pinned_weights_missing")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        binary = Path("/opt/bonsai-runtime-cpu/llama-server" if layers == "0" else "/opt/bonsai-runtime/llama-server")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(binary.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [str(binary), "-m", str(path / MODEL_FILE), "--mmproj", str(path / VISION_FILE),
               "--host", "127.0.0.1", "--port", str(port), "--alias", MODEL_ID,
               "-ngl", layers, "-fa", "on", "-c", "16384", "-np", "1", "--jinja",
               "--reasoning-budget", "0", "--reasoning-format", "auto",
               "--chat-template-kwargs", '{"enable_thinking":false}', "--no-webui"]
        # Logs contain engine state only; the adapter never logs evidence/prompt bodies.
        self.process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)
        if layers != '0':
            self.gpu_samples = []
            self.sampling_stop.clear()
            self.sampling_thread = threading.Thread(target=self._sample_gpu, daemon=True)
            self.sampling_thread.start()
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("bonsai_runtime_exited_during_load")
                try:
                    with self.http.open(self.url + "/health", timeout=2) as r:
                        if r.status == 200:
                            return
                except (OSError, URLError):
                    pass
                time.sleep(.25)
            raise TimeoutError("bonsai_runtime_load_timeout")
        except BaseException:
            self.unload()
            raise

    def unload(self):
        self.sampling_stop.set()
        if self.sampling_thread is not None:
            self.sampling_thread.join(timeout=3)
            self.sampling_thread = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        self.url = None

    @staticmethod
    def _abstain(reason):
        return ConsensusJudgeVerdict(None, 0.0, (reason,),
            {"selected_candidate_id": None, "confidence": 0.0, "reason_codes": [reason]})

    def adjudicate(self, *, batch, stage_role, candidates, comparison_payload, correction=False):
        from PIL import Image
        from fireviewer_vision_runtime.transformers_adapters import QwenConsensusJudgeAdapter

        if self.process is None or self.process.poll() is not None or not self.url:
            raise RuntimeError("bonsai_judge_not_loaded")
        if not candidates or len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("distinct_existing_candidates_required")
        sources = []
        for item in batch.items:
            if item.frames:
                sources.extend((f.frame_id, str(f.working_file_url)) for f in item.frames)
            elif item.working_file_url is not None and item.media_type in {"image", "satellite_image"}:
                sources.append((item.input_id, str(item.working_file_url)))
        maximum = int(os.getenv("FW_BONSAI_JUDGE_MAX_IMAGES", "8"))
        if not 1 <= maximum <= 8:
            raise ValueError("bonsai_image_window_must_be_between_one_and_eight")
        reason = None
        if stage_role == "asr":
            reason = "raw_audio_not_supported_by_bonsai_judge"
        elif len(sources) > maximum:
            reason = "judge_image_window_incomplete"
        elif stage_role in {"fire_detection", "visual_grounding"} and not sources:
            reason = "raw_visual_evidence_missing"
        elif not sources and not any(item.article_text for item in batch.items):
            reason = "direct_evidence_missing"
        if reason:
            self.last_receipt = {"model_called": False, "reason": reason, "source_images": len(sources)}
            return self._abstain(reason)
        content = []
        image_receipts = []
        pixel_limit = max(1, 4194304 // max(1, len(sources)))
        for evidence_id, url in sources:
            # Reuse the existing MediaFetcher host allowlist and byte limits.
            with self.fetcher.download(url) as image_path:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                try:
                    original = image.size
                    if image.width * image.height > pixel_limit:
                        factor = (pixel_limit / (image.width * image.height))**.5
                        image.thumbnail((max(1, int(image.width*factor)), max(1, int(image.height*factor))), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=95)
                    content.extend([{"type": "text", "text": "Evidence image id: " + evidence_id},
                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()}}])
                    image_receipts.append({"evidence_id": evidence_id, "original_size": list(original), "provided_size": list(image.size)})
                finally:
                    image.close()
        context = {
            "stage_role": stage_role, "source_image_count": len(sources), "provided_image_count": len(sources),
            "source_coverage_complete": True,
            "items": [{"input_id": item.input_id, "media_type": item.media_type, "article_text": item.article_text,
                       "source_context": item.source_context.model_dump(mode="json", exclude_none=True) if item.source_context else None}
                      for item in batch.items],
            "candidates": [{"candidate_id": c.candidate_id, "output_payload": c.output_payload} for c in candidates],
            "deterministic_comparison": comparison_payload, "correction": correction,
        }
        text = json.dumps(context, ensure_ascii=False, allow_nan=False)
        if len(text.encode()) > 48000:
            self.last_receipt = {"model_called": False, "reason": "judge_context_too_large"}
            return self._abstain("judge_context_too_large")
        content.append({"type": "text", "text": text})
        ids = [c.candidate_id for c in candidates]
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"selected_candidate_id": {"enum": [None, *ids]},
                                 "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                 "reason_codes": {"type": "array", "minItems": 1, "maxItems": 8,
                                                  "items": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_]{0,63}$"}}},
                  "required": ["selected_candidate_id", "confidence", "reason_codes"]}
        payload = {"model": MODEL_ID, "messages": [
            {"role": "system", "content": QwenConsensusJudgeAdapter.SYSTEM_PROMPT},
            {"role": "user", "content": content}],
            "temperature": 0, "seed": 17, "max_tokens": 256, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "consensus_verdict", "strict": True, "schema": schema}}}
        started = time.perf_counter()
        request = Request(self.url + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        try:
            with self.http.open(request, timeout=180) as response:
                raw = response.read(256001)
            if len(raw) > 256000:
                raise ValueError("response_too_large")
            data = json.loads(raw)
            answer = data["choices"][0]
            if answer.get("finish_reason") != "stop":
                raise ValueError("incomplete_bonsai_verdict")
            verdict = QwenConsensusJudgeAdapter._parse_verdict(answer["message"]["content"], candidate_ids=frozenset(ids))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ModelOutputError("bonsai_judge_invalid_or_unavailable_verdict") from exc
        self.last_receipt = {"model_called": True, "model_id": MODEL_ID, "revision": MODEL_REVISION,
                             "latency_ms": round((time.perf_counter()-started)*1000, 3),
                             "usage": data.get("usage"), "images": image_receipts,
                             "runtime": "PrismML llama.cpp prism-b10709-9a9394a", "format": "PTQ1_0",
                             "gpu_memory": self._gpu_receipt(),
                             "cpu_validation": os.getenv("FW_BONSAI_CPU_VALIDATION") == "1"}
        return ConsensusJudgeVerdict(verdict.selected_candidate_id, verdict.confidence, verdict.reason_codes,
                                     {**verdict.output_payload, "runtime_receipt": self.last_receipt})
