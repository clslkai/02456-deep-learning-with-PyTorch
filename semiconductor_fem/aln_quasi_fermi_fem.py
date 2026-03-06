#!/usr/bin/env python3
"""一维 AlN 宽禁带半导体准费米能级有限元求解器（纯标准库实现）。"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass


Q = 1.602176634e-19
K_B = 1.380649e-23


@dataclass
class AlNParameters:
    thickness_m: float = 2e-6
    voltage_v: float = 4.0
    temperature_k: float = 300.0
    n_i_m3: float = 1e10
    mobility_n_m2_vs: float = 5e-3
    mobility_p_m2_vs: float = 1e-3
    tau_n_s: float = 2e-9
    tau_p_s: float = 1e-9
    uv_generation_m3s: float = 8e30
    alpha_m_inv: float = 2e6
    elements: int = 300


class OneDFEMDriftDiffusion:
    def __init__(self, params: AlNParameters):
        self.p = params
        self.vt = K_B * params.temperature_k / Q
        self.n_nodes = params.elements + 1
        self.z = [i * params.thickness_m / params.elements for i in range(self.n_nodes)]
        self.h = params.thickness_m / params.elements
        self.d_n = self.vt * params.mobility_n_m2_vs
        self.d_p = self.vt * params.mobility_p_m2_vs

    def generation(self, z: float) -> float:
        return self.p.uv_generation_m3s * math.exp(-self.p.alpha_m_inv * z)

    def _assemble_tridiag(self, diffusion: float, tau: float):
        n = self.n_nodes
        lower = [0.0] * (n - 1)
        diag = [0.0] * n
        upper = [0.0] * (n - 1)
        rhs = [0.0] * n

        for e in range(n - 1):
            h = self.h
            z0 = self.z[e]
            z1 = self.z[e + 1]
            g0 = self.generation(z0)
            g1 = self.generation(z1)

            k00 = diffusion / h + h / (3.0 * tau)
            k01 = -diffusion / h + h / (6.0 * tau)
            k11 = diffusion / h + h / (3.0 * tau)

            f0 = h / 6.0 * (2.0 * g0 + g1)
            f1 = h / 6.0 * (g0 + 2.0 * g1)

            diag[e] += k00
            upper[e] += k01
            lower[e] += k01
            diag[e + 1] += k11
            rhs[e] += f0
            rhs[e + 1] += f1

        return lower, diag, upper, rhs

    @staticmethod
    def _apply_dirichlet_tridiag(lower, diag, upper, rhs, left_value, right_value):
        n = len(diag)

        rhs[1] -= lower[0] * left_value
        lower[0] = 0.0
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = left_value

        rhs[n - 2] -= upper[n - 2] * right_value
        upper[n - 2] = 0.0
        diag[n - 1] = 1.0
        lower[n - 2] = 0.0
        rhs[n - 1] = right_value

        return lower, diag, upper, rhs

    @staticmethod
    def _solve_tridiagonal(lower, diag, upper, rhs):
        n = len(diag)
        c = upper[:]
        d = rhs[:]
        b = diag[:]

        for i in range(1, n):
            if abs(b[i - 1]) < 1e-300:
                raise ZeroDivisionError("三对角求解出现零主元。")
            w = lower[i - 1] / b[i - 1]
            b[i] -= w * c[i - 1] if i - 1 < len(c) else 0.0
            d[i] -= w * d[i - 1]

        x = [0.0] * n
        x[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

        return x

    def _solve_carrier(self, diffusion: float, tau: float, left: float, right: float):
        lower, diag, upper, rhs = self._assemble_tridiag(diffusion, tau)
        lower, diag, upper, rhs = self._apply_dirichlet_tridiag(lower, diag, upper, rhs, left, right)
        return self._solve_tridiagonal(lower, diag, upper, rhs)

    def solve(self):
        n_i = self.p.n_i_m3

        n_left = n_i
        p_left = n_i
        n_right = n_i * math.exp(self.p.voltage_v / self.vt)
        p_right = n_i * math.exp(-self.p.voltage_v / self.vt)

        n = self._solve_carrier(self.d_n, self.p.tau_n_s, n_left, n_right)
        p = self._solve_carrier(self.d_p, self.p.tau_p_s, p_left, p_right)

        n = [max(val, 1e-30) for val in n]
        p = [max(val, 1e-30) for val in p]

        e_fn = [self.vt * math.log(val / n_i) for val in n]
        e_fp = [-self.vt * math.log(val / n_i) for val in p]

        return {
            "z_m": self.z,
            "n_m3": n,
            "p_m3": p,
            "E_fn_eV": e_fn,
            "E_fp_eV": e_fp,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlN 一维准费米能级 FEM 求解")
    parser.add_argument("--thickness-um", type=float, default=2.0, help="材料厚度 (um)")
    parser.add_argument("--voltage", type=float, default=4.0, help="施加电势 (V)")
    parser.add_argument("--elements", type=int, default=300, help="有限元单元数")
    parser.add_argument("--generation", type=float, default=8e30, help="表面体生成率 G0 (m^-3 s^-1)")
    parser.add_argument("--alpha", type=float, default=2e6, help="吸收系数 alpha (m^-1)")
    parser.add_argument("--output", type=str, default="aln_quasi_fermi.csv", help="输出 CSV 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = AlNParameters(
        thickness_m=args.thickness_um * 1e-6,
        voltage_v=args.voltage,
        elements=args.elements,
        uv_generation_m3s=args.generation,
        alpha_m_inv=args.alpha,
    )

    solver = OneDFEMDriftDiffusion(params)
    result = solver.solve()

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["z_m", "n_m3", "p_m3", "E_fn_eV", "E_fp_eV"])
        for i in range(len(result["z_m"])):
            writer.writerow([
                result["z_m"][i],
                result["n_m3"][i],
                result["p_m3"][i],
                result["E_fn_eV"][i],
                result["E_fp_eV"][i],
            ])

    print("求解完成。")
    print(f"厚度 = {params.thickness_m * 1e6:.3f} um, 电势 = {params.voltage_v:.3f} V, 单元数 = {params.elements}")
    print(f"结果写入: {args.output}")
    print(f"E_fn 范围: [{min(result['E_fn_eV']):.4e}, {max(result['E_fn_eV']):.4e}] eV")
    print(f"E_fp 范围: [{min(result['E_fp_eV']):.4e}, {max(result['E_fp_eV']):.4e}] eV")


if __name__ == "__main__":
    main()
