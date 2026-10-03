"""Artifact preparation contracts, without a GPU, network access or model downloads."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import ornith_fetch
from test_hfmodel_ornith import put_snapshot

class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.cache = Path(self.tmp.name)
        self.calls = []
    def tearDown(self): self.tmp.cleanup()
    def download(self,**kwargs):
        self.calls.append(kwargs)
        return str(put_snapshot(self.cache,kwargs["repo_id"],kwargs["revision"] or "main-rev",kwargs["filename"]))
    def fetch(self,**kwargs):
        return ornith_fetch.fetch(self.cache,download=self.download,free_bytes=100<<30,**kwargs)
    def test_first_run_fetches_exact_main_and_separate_mtp(self):
        result = self.fetch(spec=3)
        self.assertEqual([x["filename"] for x in self.calls],
                         ["Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf","mtpdraft-Q8_0.gguf"])
        self.assertNotEqual(self.calls[0]["repo_id"],self.calls[1]["repo_id"])
        self.assertTrue(all(Path(x).is_file() for x in result.values()))
    def test_cached_start_uses_no_download(self):
        first = self.fetch(spec=2); self.calls.clear()
        self.assertEqual(self.fetch(spec=2,offline=True),first); self.assertEqual(self.calls,[])
    def test_spec_off_never_fetches_mtp(self):
        self.assertEqual(self.fetch()["MTP_GGUF"],""); self.assertEqual(len(self.calls),1)
    def test_offline_missing_fails_before_download(self):
        with self.assertRaisesRegex(ValueError,"offline"):
            self.fetch(spec=2,offline=True)
        self.assertEqual(self.calls,[])
    def test_space_floor_is_enforced_before_download(self):
        with self.assertRaisesRegex(ValueError,"20 GiB floor"):
            ornith_fetch.fetch(self.cache,spec=2,download=self.download,free_bytes=30<<30)
        self.assertEqual(self.calls,[])
    def test_direct_artifacts_bypass_cache(self):
        main = self.cache/"direct.gguf"; draft=self.cache/"draft.gguf"
        main.write_bytes(b"GGUF"); draft.write_bytes(b"GGUF")
        result = self.fetch(native=str(main),mtp=str(draft),spec=2,offline=True)
        self.assertEqual(result,{"NATIVE":str(main),"MTP_GGUF":str(draft)})
        self.assertEqual(self.calls,[])
    def test_explicit_missing_file_is_an_error(self):
        with self.assertRaisesRegex(ValueError,"explicit artifact"):
            self.fetch(native=str(self.cache/"missing.gguf"))
        self.assertEqual(self.calls,[])
    def test_mtp_revision_is_independent(self):
        with patch.dict(os.environ,{"STRATA_MTP_REV":"draft-revision"}):
            self.fetch(spec=2,revision="main-revision")
        self.assertEqual([x["revision"] for x in self.calls],["main-revision","draft-revision"])
    def test_partial_download_is_not_reused(self):
        def failed(**kwargs):
            path = self.cache/"partial.incomplete"; path.write_bytes(b"partial")
            raise OSError("interrupted download")
        with self.assertRaisesRegex(OSError,"interrupted"):
            ornith_fetch.fetch(self.cache,download=failed,free_bytes=100<<30)
        self.fetch(); self.assertEqual(len(self.calls),1)

if __name__ == "__main__": unittest.main()
