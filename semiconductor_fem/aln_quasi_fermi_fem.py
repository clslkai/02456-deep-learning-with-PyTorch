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


def _polyline_points(xs, ys, x_min, x_max, y_min, y_max, width, height, margin):
    x0, y0, x1, y1 = margin, margin, width - margin, height - margin
    pts = []
    for x, y in zip(xs, ys):
        xp = x0 + (x - x_min) / (x_max - x_min) * (x1 - x0)
        yp = y1 - (y - y_min) / (y_max - y_min) * (y1 - y0)
        pts.append(f"{xp:.2f},{yp:.2f}")
    return " ".join(pts)


def write_fermi_plot_svg(result, thickness_um, output_svg, voltage_v, cbm0_ev=0.35, vbm0_ev=-0.95):
    z_m = result["z_m"]
    x_um = [z * 1e6 for z in z_m]
    l_um = max(thickness_um, 1e-15)

    # 用线性电势降构造示意能带边（和样图风格接近）
    delta_ev = max(min(voltage_v * 0.06, 0.6), -0.6)
    e_cbm = [cbm0_ev + delta_ev * (x / l_um) for x in x_um]
    e_vbm = [vbm0_ev + delta_ev * (x / l_um) for x in x_um]

    # 费米能级用电子/空穴准费米的中线表示
    e_fermi = [(a + b) * 0.5 for a, b in zip(result["E_fn_eV"], result["E_fp_eV"])]

    y_all = e_cbm + e_vbm + e_fermi
    y_min = min(y_all) - 0.08
    y_max = max(y_all) + 0.08
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    width, height = 900, 560
    margin = 70

    cbm_pts = _polyline_points(x_um, e_cbm, 0.0, l_um, y_min, y_max, width, height, margin)
    vbm_pts = _polyline_points(x_um, e_vbm, 0.0, l_um, y_min, y_max, width, height, margin)
    ef_pts = _polyline_points(x_um, e_fermi, 0.0, l_um, y_min, y_max, width, height, margin)

    x0, y0, x1, y1 = margin, margin, width - margin, height - margin

    def y_to_px(v):
        return y1 - (v - y_min) / (y_max - y_min) * (y1 - y0)

    xticks = [0.0, l_um * 0.25, l_um * 0.5, l_um * 0.75, l_um]
    yticks = [y_min + i * (y_max - y_min) / 6.0 for i in range(7)]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="#f2f2f2"/>')

    for t in xticks:
        x = x0 + (t / l_um) * (x1 - x0)
        svg.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="#d8d8d8" stroke-width="1"/>')
    for t in yticks:
        y = y_to_px(t)
        svg.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#d8d8d8" stroke-width="1"/>')

    svg.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#404040" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#404040" stroke-width="2"/>')

    svg.append(f'<polyline points="{cbm_pts}" fill="none" stroke="#5b6cff" stroke-width="3"/>')
    svg.append(f'<polyline points="{ef_pts}" fill="none" stroke="#333333" stroke-width="2.5" stroke-dasharray="8,6"/>')
    svg.append(f'<polyline points="{vbm_pts}" fill="none" stroke="#54d66b" stroke-width="3"/>')

    legend_x, legend_y = x1 - 170, y0 + 20
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="145" height="95" fill="#ffffff" stroke="#333"/>')
    svg.append(f'<line x1="{legend_x + 10}" y1="{legend_y + 20}" x2="{legend_x + 40}" y2="{legend_y + 20}" stroke="#5b6cff" stroke-width="3"/>')
    svg.append(f'<text x="{legend_x + 45}" y="{legend_y + 25}" font-size="22" fill="#222">CBM</text>')
    svg.append(f'<line x1="{legend_x + 10}" y1="{legend_y + 47}" x2="{legend_x + 40}" y2="{legend_y + 47}" stroke="#333" stroke-width="2.5" stroke-dasharray="8,6"/>')
    svg.append(f'<text x="{legend_x + 45}" y="{legend_y + 52}" font-size="22" fill="#222">E_Fermi</text>')
    svg.append(f'<line x1="{legend_x + 10}" y1="{legend_y + 74}" x2="{legend_x + 40}" y2="{legend_y + 74}" stroke="#54d66b" stroke-width="3"/>')
    svg.append(f'<text x="{legend_x + 45}" y="{legend_y + 79}" font-size="22" fill="#222">VBM</text>')

    for t in xticks:
        x = x0 + (t / l_um) * (x1 - x0)
        svg.append(f'<text x="{x:.2f}" y="{y1 + 34}" text-anchor="middle" font-size="22" fill="#333">{t:.3g}</text>')
    for t in yticks:
        y = y_to_px(t)
        svg.append(f'<text x="{x0 - 12}" y="{y + 7:.2f}" text-anchor="end" font-size="22" fill="#333">{t:.2f}</text>')

    svg.append(f'<text x="{(x0 + x1) * 0.5}" y="{height - 12}" text-anchor="middle" font-size="36" fill="#333">Thickness (μm)</text>')
    svg.append(f'<text x="24" y="{(y0 + y1) * 0.5}" text-anchor="middle" font-size="36" fill="#333" transform="rotate(-90 24 {(y0 + y1) * 0.5})">Energy (eV)</text>')

    svg.append('</svg>')

    with open(output_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlN 一维准费米能级 FEM 求解")
    parser.add_argument("--thickness-um", type=float, default=2.0, help="材料厚度 (um)")
    parser.add_argument("--voltage", type=float, default=4.0, help="施加电势 (V)")
    parser.add_argument("--elements", type=int, default=300, help="有限元单元数")
    parser.add_argument("--generation", type=float, default=8e30, help="表面体生成率 G0 (m^-3 s^-1)")
    parser.add_argument("--alpha", type=float, default=2e6, help="吸收系数 alpha (m^-1)")
    parser.add_argument("--output", type=str, default="aln_quasi_fermi.csv", help="输出 CSV 路径")
    parser.add_argument("--plot", type=str, default="aln_fermi_profile.svg", help="输出费米能级分布图 (SVG)")
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

    write_fermi_plot_svg(
        result=result,
        thickness_um=args.thickness_um,
        output_svg=args.plot,
        voltage_v=args.voltage,
    )

    print("求解完成。")
    print(f"厚度 = {params.thickness_m * 1e6:.3f} um, 电势 = {params.voltage_v:.3f} V, 单元数 = {params.elements}")
    print(f"结果写入: {args.output}")
    print(f"图像写入: {args.plot}")
    print(f"E_fn 范围: [{min(result['E_fn_eV']):.4e}, {max(result['E_fn_eV']):.4e}] eV")
    print(f"E_fp 范围: [{min(result['E_fp_eV']):.4e}, {max(result['E_fp_eV']):.4e}] eV")


if __name__ == "__main__":
    main()
