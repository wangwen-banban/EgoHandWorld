"""Unit and smoke tests for EgoHandWorld core and object_hoi extension."""
import ast
import py_compile
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJ_DIR = ROOT / "object_hoi"


class TestObjectHOI(unittest.TestCase):
    def test_all_python_files_compile(self):
        py_files = list(OBJ_DIR.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 5, "Expected core object_hoi scripts")
        for p in py_files:
            with self.subTest(file=p.name):
                py_compile.compile(str(p), doraise=True)
                ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

    def test_no_internal_ips_or_tokens(self):
        pattern = re.compile(r"10\.29\.|10\.208\.|gho_[A-Za-z0-9]+|hf_[A-Za-z0-9]+")
        for p in list(OBJ_DIR.glob("*")) + [ROOT / "README.md"]:
            if p.is_file():
                txt = p.read_text(encoding="utf-8", errors="ignore")
                self.assertIsNone(
                    pattern.search(txt),
                    f"Found private IP or token in {p.relative_to(ROOT)}",
                )

    def test_required_assets_present(self):
        for fname in (
            "brush_bowl_cmp_2x2.gif",
            "cut_knife_plate_cmp_2x2.gif",
            "benchmark_6seq_bars.png",
            "jitter_and_depth_curves.png",
        ):
            p = ROOT / "assets" / fname
            self.assertTrue(p.exists() and p.stat().st_size > 10_000, f"Missing or empty asset: {fname}")

    def test_foundationpose_supports_vipe_intrinsics_and_hand_npz(self):
        src = (OBJ_DIR / "run_foundationpose.py").read_text(encoding="utf-8")
        self.assertIn("--hand_npz", src)
        self.assertIn("K_use", src)


if __name__ == "__main__":
    unittest.main()
