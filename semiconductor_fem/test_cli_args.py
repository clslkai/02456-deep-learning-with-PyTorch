import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestAlnQuasiFermiCLI(unittest.TestCase):
    def test_default_equilibrium_has_flat_fermi(self):
        script = Path(__file__).with_name("aln_quasi_fermi_fem.py")
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "eq.csv"
            out_svg = Path(tmp) / "eq.svg"
            cmd = [
                "python", str(script),
                "--thickness-um", "0.05",
                "--voltage", "4.0",
                "--elements", "50",
                "--bandgap-ev", "6.2",
                "--output", str(out_csv),
                "--plot", str(out_svg),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            with out_csv.open("r", encoding="utf-8") as f:
                r = csv.reader(f)
                header = next(r)
                rows = list(r)
            self.assertEqual(header, ["z_m", "n_m3", "p_m3", "E_fn_eV", "E_fp_eV"])
            efn = [float(row[3]) for row in rows]
            efp = [float(row[4]) for row in rows]
            self.assertAlmostEqual(max(efn) - min(efn), 0.0, places=9)
            self.assertAlmostEqual(max(efp) - min(efp), 0.0, places=9)
            self.assertAlmostEqual(efn[0], 3.1, places=6)
            self.assertIn("E_F (flat)", out_svg.read_text(encoding="utf-8"))

    def test_quasi_fermi_mode_has_split_labels(self):
        script = Path(__file__).with_name("aln_quasi_fermi_fem.py")
        with tempfile.TemporaryDirectory() as tmp:
            out_svg = Path(tmp) / "qf.svg"
            out_csv = Path(tmp) / "qf.csv"
            cmd = [
                "python", str(script),
                "--mode", "quasi-fermi",
                "--output", str(out_csv),
                "--plot", str(out_svg),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            svg = out_svg.read_text(encoding="utf-8")
            self.assertIn("E_Fn", svg)
            self.assertIn("E_Fp", svg)


if __name__ == "__main__":
    unittest.main()
