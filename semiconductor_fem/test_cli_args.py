import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestAlnQuasiFermiCLI(unittest.TestCase):
    def test_plot_argument_is_supported_and_outputs_files(self):
        script = Path(__file__).with_name("aln_quasi_fermi_fem.py")
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "out.csv"
            out_svg = Path(tmp) / "out.svg"

            cmd = [
                "python",
                str(script),
                "--thickness-um",
                "0.05",
                "--voltage",
                "4.0",
                "--elements",
                "50",
                "--output",
                str(out_csv),
                "--plot",
                str(out_svg),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(out_csv.exists(), "CSV output was not created")
            self.assertTrue(out_svg.exists(), "SVG output was not created")

            with out_csv.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
            self.assertEqual(header, ["z_m", "n_m3", "p_m3", "E_fn_eV", "E_fp_eV"])

            svg_text = out_svg.read_text(encoding="utf-8")
            self.assertIn("CBM", svg_text)
            self.assertIn("E_Fermi", svg_text)
            self.assertIn("VBM", svg_text)


if __name__ == "__main__":
    unittest.main()
