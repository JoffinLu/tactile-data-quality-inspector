"""tactile-qc 合成数据 demo —— 无需 RCT 数据集即可体验完整流水线。

生成一个 48 条序列的小型合成触觉数据集（4 个材料类别，含已知缺陷注入），
然后跑完整的质量评分 + 异常检测流水线，并把结果打印到终端、
导出 CSV 到 ``examples/demo_output/``。

用法::

    # 已安装本包（pip install -e . 或 pip install tactile-qc）
    python examples/run_demo.py

    # 未安装时，从仓库根目录：
    python examples/run_demo.py --root ..
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tactile_qc.anomaly import detect_anomalies
from tactile_qc.quality import compute_quality_score

# --------------------------------------------------------------------------- #
# 1. 合成数据生成
# --------------------------------------------------------------------------- #
MATERIALS = ("Plastic_Rubber", "Metal", "Paper_Cardboard", "Textiles_Leather")

# 每条序列的缺陷类型（None = 正常序列）
DEFECTS = {
    5: "heavy_noise",       # 传感器噪声异常
    11: "saturated",        # 过曝/饱和帧
    17: "baseline_drift",   # 基线漂移
    23: "force_spike",      # 力信号突变
    29: "heavy_noise",
    35: "saturated",
    41: "baseline_drift",
    47: "force_spike",
}


def _make_frames(rng: np.random.Generator, t: int, size: int = 32,
                 base: float = 120.0, noise: float = 4.0,
                 drift: float = 0.0, saturated: bool = False) -> np.ndarray:
    """一条 (T, H, W) uint8 合成触觉序列：高斯亮斑 + 噪声 + 可选缺陷。"""
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    # 一个中心受压亮斑，随深度增强 —— 模拟 DIGIT 接触印迹
    depth = np.linspace(0.3, 1.0, t)[:, None, None]
    r2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
    blob = 60.0 * np.exp(-r2 / (2 * (size / 5) ** 2))[None, :, :] * depth
    idx = np.arange(t)[:, None, None]
    frames = base + blob + drift * idx + rng.normal(0, noise, (t, h, w))
    if saturated:
        frames = frames + 110.0  # 推到接近 255 -> 大面积饱和
    return np.clip(frames, 0, 255).astype(np.uint8)


def _make_forces(rng: np.random.Generator, t: int, spike: bool) -> np.ndarray:
    """(T, 6) 力/力矩轨迹：Fz 随深度平滑上升，可选注入突变。"""
    z = np.linspace(0.0, 1.6, t)
    fz = 0.5 + 4.0 * z + rng.normal(0, 0.05, t)
    if spike and t > 4:
        fz[t // 2] += 25.0  # 瞬时冲击
        fz[t // 2 + 1] -= 8.0
    forces = np.zeros((t, 6))
    forces[:, 2] = fz
    return forces


def make_synthetic_dataset(n_per_material: int = 12, seed: int = 0):
    """生成带已知缺陷的合成序列列表（duck-typed，同 io.TactileSequence 接口）。"""
    rng = np.random.default_rng(seed)
    seqs = []
    for i in range(n_per_material * len(MATERIALS)):
        material = MATERIALS[i % len(MATERIALS)]
        defect = DEFECTS.get(i)
        t = int(rng.integers(12, 20))
        frames = _make_frames(
            rng, t,
            noise=30.0 if defect == "heavy_noise" else 4.0,
            drift=3.0 if defect == "baseline_drift" else 0.0,
            saturated=(defect == "saturated"),
        )
        forces = _make_forces(rng, t, spike=(defect == "force_spike"))
        seqs.append(
            SimpleNamespace(
                sequence_id=f"synth_{i:03d}",
                material_label=material,
                has_force=True,
                forces=forces,
                z_positions=np.linspace(0.0, 1.6, t) - 0.2,  # 覆盖接近段+接触段
                depth_values=np.linspace(0.0, 1.6, t),
                load_frames=lambda fr=frames: fr,
            )
        )
    return seqs


# --------------------------------------------------------------------------- #
# 2. 跑流水线 + 汇报
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="tactile-qc synthetic demo")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "demo_output",
                        help="结果输出目录（默认 examples/demo_output/）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 68)
    print("tactile-qc demo — 合成数据流水线（无需 RCT 数据集）")
    print("=" * 68)

    dataset = make_synthetic_dataset(seed=args.seed)
    n_defect = sum(1 for i in DEFECTS if i < len(dataset))
    print(f"\n[1/3] 合成数据集：{len(dataset)} 条序列 × "
          f"{len(MATERIALS)} 个材料类别，注入 {n_defect} 条已知缺陷序列")
    for d in ("heavy_noise", "saturated", "baseline_drift", "force_spike"):
        ids = [f"synth_{i:03d}" for i, v in DEFECTS.items()
               if v == d and i < len(dataset)]
        print(f"    - {d:14s}: {', '.join(ids)}")

    print("\n[2/3] 质量评分（SNR / 基线漂移 / 饱和率 / 力异常 / SPC）")
    qdf = compute_quality_score(dataset)
    good = qdf[qdf["quality_score"] >= qdf["quality_score"].median()]
    print(f"    质量分中位数 {qdf['quality_score'].median():.1f} / "
          f"均值 {qdf['quality_score'].mean():.1f}（0-100，相对评分）")
    worst = qdf.nsmallest(5, "quality_score")
    print("    质量分最低 5 条：")
    for _, r in worst.iterrows():
        print(f"      {r['sequence_id']}  score={r['quality_score']:5.1f}  "
              f"snr={r['snr']:6.1f}dB  sat={r['saturation_ratio']:.2f}")

    print("\n[3/3] 异常检测（IsolationForest + Mahalanobis，无监督）")
    rep = detect_anomalies(dataset, contamination=n_defect / len(dataset),
                           pca_components=8, img_size=16)
    flagged = rep.per_sequence[rep.per_sequence["iso_label"] == -1]
    print(f"    IsolationForest 标记 {len(flagged)} 条（期望约 {n_defect} 条缺陷）")
    truth = {f"synth_{i:03d}" for i in DEFECTS if i < len(dataset)}
    hit = truth & set(flagged["sequence_id"])
    if truth:
        print(f"    已知缺陷命中率：{len(hit)}/{len(truth)} "
              f"（{len(hit) / len(truth):.0%}）")
    print(f"    两方法一致性 Cohen's kappa = {rep.comparison.cohen_kappa:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    q_path = args.output_dir / "demo_quality_scores.csv"
    a_path = args.output_dir / "demo_anomaly_results.csv"
    qdf.to_csv(q_path, index=False, encoding="utf-8-sig")
    rep.per_sequence.to_csv(a_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已导出：\n  {q_path}\n  {a_path}")
    print("\ndemo 完成。真实数据用法见 README：tactile-qc run --data-dir <RCT目录>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
