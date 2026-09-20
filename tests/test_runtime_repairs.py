import json, unittest
from fireviewer_vision_runtime.transformers_adapters import QwenAdapter
class FormatNormalization(unittest.TestCase):
 def test_one_exact_json_fence(self):
  p={'observations':[{'type':'smoke_visible','evidence_kind':'image','evidence_id':'x','description':'smoke','certainty':'directly_visible'}],'explicit_places':[],'explicit_times':[]}
  a=QwenAdapter._parse(json.dumps(p));b=QwenAdapter._parse('```json\n'+json.dumps(p)+'\n```')
  self.assertEqual(a,b);self.assertEqual(len(b[0]),1)
 def test_surrounding_prose_is_rejected(self):
  with self.assertRaises(ValueError):QwenAdapter._parse('Here is the answer: {"observations":[],"explicit_places":[],"explicit_times":[]}')
 def test_invalid_certainty_is_still_rejected(self):
  p={'observations':[{'type':'smoke','evidence_kind':'image','evidence_id':'x','description':'smoke','certainty':'certain'}],'explicit_places':[],'explicit_times':[]}
  with self.assertRaises(ValueError):QwenAdapter._parse('```json\n'+json.dumps(p)+'\n```')
if __name__=='__main__':unittest.main()
