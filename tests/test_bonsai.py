import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fireviewer_contracts.contracts import WorkerInput
from fireviewer_contracts.model_registry import CONSENSUS_JUDGE, ModelSpec, build_registry
from fireviewer_contracts.mvp_stack import load_mvp_stack
from fireviewer_vision_runtime.bonsai_judge import BonsaiConsensusJudgeAdapter
from fireviewer_vision_runtime.consensus import JudgeCandidate


class BonsaiBoundaryTests(unittest.TestCase):
    def make(self):
        return BonsaiConsensusJudgeAdapter(CONSENSUS_JUDGE, cache_root=Path('/missing'), fetcher=None)

    def test_profile_keeps_research_and_human_validation(self):
        stack = load_mvp_stack()
        self.assertEqual(stack.hardware.vram_gib, 24)
        self.assertEqual(stack.status, 'experimental')
        self.assertEqual(build_registry()['source_research'].model_id, 'Qwen/Qwen3-14B')
        self.assertFalse(stack.auto_publication)
        self.assertTrue(stack.human_validation_required)

    def test_model_pin_required(self):
        wrong = ModelSpec(role='consensus_judge', model_id=CONSENSUS_JUDGE.model_id, revision='0'*40)
        with self.assertRaises(ValueError):
            BonsaiConsensusJudgeAdapter(wrong, cache_root=Path('/missing'), fetcher=None)

    def test_cpu_fallback_forbidden_without_validation_opt_in(self):
        with patch.dict(os.environ, {'FW_BONSAI_GPU_LAYERS': '0', 'FW_BONSAI_CPU_VALIDATION': ''}):
            with self.assertRaisesRegex(RuntimeError, 'cpu_mode_only'):
                self.make().load()

    def test_missing_pixels_and_incomplete_windows_abstain_without_http(self):
        judge = self.make()
        judge.process = SimpleNamespace(poll=lambda: None)
        judge.url = 'http://127.0.0.1:1'
        candidates = (JudgeCandidate('a', 'test', 'test', {}), JudgeCandidate('b', 'test', 'test', {}))
        for items, reason in [
            ([{'input_id': 'text', 'media_type': 'article', 'article_text': 'Smoke reported.'}], 'raw_visual_evidence_missing'),
            ([{'input_id': 'image_'+str(i), 'media_type': 'image', 'working_file_url': 'https://evidence.example/image.jpg'} for i in range(9)], 'judge_image_window_incomplete')]:
            batch = WorkerInput.model_validate({'schema_version': '1.0', 'batch_id': 'probe', 'batch_type': 'external_media',
                'priority': 'scheduled', 'items': items})
            verdict = judge.adjudicate(batch=batch, stage_role='fire_detection', candidates=candidates, comparison_payload={})
            self.assertIsNone(verdict.selected_candidate_id)
            self.assertIn(reason, verdict.reason_codes)
            self.assertFalse(judge.last_receipt['model_called'])

    def test_no_gpu_measurement_is_not_zero(self):
        judge = self.make()
        self.assertEqual(judge._gpu_receipt(), {'status': 'UNMEASURED'})


if __name__ == '__main__':
    unittest.main()
