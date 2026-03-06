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
                "--bandgap-ev",
                "6.2",
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
                first_row = next(reader)
            self.assertEqual(header, ["z_m", "n_m3", "p_m3", "E_fn_eV", "E_fp_eV"])

            # z=0 端为本征边界，费米能级应在禁带中心 Eg/2=3.1 eV 附近
            efn0 = float(first_row[3])
            efp0 = float(first_row[4])
            self.assertAlmostEqual(efn0, 3.1, places=6)
            self.assertAlmostEqual(efp0, 3.1, places=6)

            svg_text = out_svg.read_text(encoding="utf-8")
            self.assertIn("CBM", svg_text)
            self.assertIn("E_Fermi", svg_text)
            self.assertIn("VBM", svg_text)


if __name__ == "__main__":
    unittest.main()
