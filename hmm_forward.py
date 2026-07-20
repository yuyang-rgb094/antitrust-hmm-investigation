#!/usr/bin/env python3
"""
反垄断调查 HMM 概率推演 — 前向算法实现

Usage:
  python hmm_forward.py                         # 使用默认参数（澜起案例）
  python hmm_forward.py --pi 0.1 0.5 0.3 0.1    # 自定义初始概率
  python hmm_forward.py --obs O1 O2 O3           # 自定义观测序列
  python hmm_forward.py --verbose                # 详细输出每步递推

状态空间:
  S1: 无果结案     S2: 和解罚款
  S3: 实质性处罚   S4: 极端连锁反应

观测空间:
  O1: 检方无新进展, 公司回应"配合调查"
  O2: 检方公布初步调查结论
  O3: 检方移交审判 / 起诉高管
  O4: 公司公告大额罚款计提
  O5: 客户启动替代供应商验证
  O6: 美国DOJ宣布调查结果/罚款
  O7: 三星/海力士/Micron发布降价/赔偿公告
"""

import argparse
import json
import sys
import numpy as np

# ──────────────────────────────────────────
# 默认参数（基于澜起科技反垄断调查案例）
# ──────────────────────────────────────────

# 状态名称
STATES = ["S1:无果结案", "S2:和解罚款", "S3:实质性处罚", "S4:极端连锁"]
N = len(STATES)

# 观测名称
OBS_NAMES = {
    "O1": "检方无新进展,公司回应配合调查",
    "O2": "检方公布初步调查结论",
    "O3": "检方移交审判/起诉高管",
    "O4": "公司公告大额罚款计提",
    "O5": "客户启动替代供应商验证",
    "O6": "美国DOJ宣布调查结果/罚款",
    "O7": "三星/海力士/Micron发布降价/赔偿公告",
}
OBS_KEYS = list(OBS_NAMES.keys())  # O1..O7

# 初始概率分布 π
DEFAULT_PI = np.array([0.15, 0.45, 0.30, 0.10])

# 状态转移矩阵 A (4×4)
DEFAULT_A = np.array([
    [0.95, 0.04, 0.01, 0.00],
    [0.05, 0.85, 0.08, 0.02],
    [0.00, 0.10, 0.80, 0.10],
    [0.00, 0.00, 0.15, 0.85],
])

# 观测矩阵 B (4×7): B[i][j] = P(O_{j+1} | S_{i+1})
DEFAULT_B = np.array([
    [0.50, 0.20, 0.02, 0.01, 0.01, 0.10, 0.16],  # S1
    [0.15, 0.35, 0.05, 0.05, 0.05, 0.10, 0.25],  # S2
    [0.02, 0.15, 0.20, 0.15, 0.15, 0.10, 0.23],  # S3
    [0.01, 0.05, 0.15, 0.10, 0.20, 0.20, 0.29],  # S4
])


def parse_observations(obs_list):
    """将 O1, O2, ... 字符串转为列索引 (0-based)."""
    indices = []
    for o in obs_list:
        o = o.upper().strip()
        if o not in OBS_KEYS:
            valid = ", ".join(OBS_KEYS)
            raise ValueError(f"未知观测 '{o}'，有效值: {valid}")
        indices.append(OBS_KEYS.index(o))
    return indices


def forward_algorithm(pi, A, B, obs_indices, verbose=False):
    """
    前向算法 (Forward Algorithm)

    参数:
        pi: 初始概率分布, shape (N,)
        A:  转移矩阵, shape (N, N)
        B:  观测矩阵, shape (N, M)
        obs_indices: 观测序列的列索引列表
        verbose: 是否打印每步递推

    返回:
        alpha_norm: 最终归一化后的概率分布 (对应 P(S | O_1..O_T))
        history:    每步的归一化概率 (list of ndarray)
    """
    T = len(obs_indices)
    history = []

    # t=1: 初始化
    alpha = pi * B[:, obs_indices[0]]
    alpha_norm = alpha / alpha.sum()
    history.append(alpha_norm.copy())

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 1  (观测: {OBS_NAMES[OBS_KEYS[obs_indices[0]]]})")
        print(f"{'='*60}")
        print("  未归一化 α₁:")
        for i, s in enumerate(STATES):
            print(f"    {s:20s}  =  {alpha[i]:.6f}")
        print(f"  归一化后:")
        for i, s in enumerate(STATES):
            print(f"    {s:20s}  =  {alpha_norm[i]*100:.2f}%")

    # t >= 2: 递推
    for t in range(1, T):
        # α_t(j) = [Σ_i α_{t-1}(i) * A_ij] * B_j(O_t)
        alpha_prev = history[-1]
        pred = alpha_prev @ A  # 预测分布
        alpha = pred * B[:, obs_indices[t]]
        alpha_norm = alpha / alpha.sum()
        history.append(alpha_norm.copy())

        if verbose:
            print(f"\n{'='*60}")
            print(f"Step {t+1}  (观测: {OBS_NAMES[OBS_KEYS[obs_indices[t]]]})")
            print(f"{'='*60}")
            print("  预测分布 (α_{t-1} @ A):")
            for i, s in enumerate(STATES):
                print(f"    {s:20s}  =  {pred[i]*100:.2f}%")
            print("  未归一化 α_t:")
            for i, s in enumerate(STATES):
                print(f"    {s:20s}  =  {alpha[i]:.6f}")
            print(f"  归一化后:")
            for i, s in enumerate(STATES):
                print(f"    {s:20s}  =  {alpha_norm[i]*100:.2f}%")

    return history[-1], history


def print_result(final_dist, history):
    """打印最终结果和评级."""
    print(f"\n{'#'*60}")
    print(f"  HMM 前向算法推演结果")
    print(f"{'#'*60}")
    print(f"\n最终概率分布（共 {len(history)} 步递推）:")
    print(f"{'-'*40}")
    for i, s in enumerate(STATES):
        pct = final_dist[i] * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {s:22s}  {pct:5.1f}%  {bar}")

    print(f"\n{'评级与应对策略':-^40}")
    if final_dist[0] > 0.40:
        print(f"  ✅ 偏乐观 — S1(无果结案)>{40:.0f}%，被错杀标的可逢低布局")
    if final_dist[1] > 0.40:
        print(f"  📊 中性 — S2(和解罚款)>{40:.0f}%，需评估罚款对财务的影响")
    if final_dist[2] > 0.20:
        print(f"  ⚠️  警惕 — S3(实质性处罚)>{20:.0f}%，需调整仓位")
    if final_dist[3] > 0.05:
        print(f"  🔴 高风险 — S4(极端连锁)>{5:.0f}%，需大幅减仓")

    # 最可能状态
    max_idx = final_dist.argmax()
    print(f"\n👉 最可能结果: {STATES[max_idx]} ({final_dist[max_idx]*100:.1f}%)")

    # 仓位建议
    base_position = 1.0  # 基准仓位
    suggested = base_position * (1 - final_dist[2] - 2 * final_dist[3])
    print(f"  仓位建议: 基准仓位 × (1 - S3概率 - 2×S4概率)")
    print(f"            = {base_position:.0%} × (1 - {final_dist[2]*100:.1f}% - 2×{final_dist[3]*100:.1f}%)")
    print(f"            = {suggested*100:.1f}% 仓位")

    # 输出JSON格式（便于程序化调用）
    print(f"\n{'JSON 输出':-^40}")
    json_out = {
        "final_distribution": {
            "S1_无果结案": round(float(final_dist[0]), 4),
            "S2_和解罚款": round(float(final_dist[1]), 4),
            "S3_实质性处罚": round(float(final_dist[2]), 4),
            "S4_极端连锁": round(float(final_dist[3]), 4),
        },
        "most_likely": STATES[max_idx],
        "suggested_position_pct": round(suggested * 100, 1),
    }
    print(json.dumps(json_out, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="反垄断调查 HMM 概率推演 — 前向算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python hmm_forward.py
  python hmm_forward.py --pi 0.2 0.5 0.2 0.1 --obs O1 O2
  python hmm_forward.py --pi 0.1 0.3 0.4 0.2 --obs O1 O2 O3 O4 --verbose
        """,
    )
    parser.add_argument(
        "--pi", nargs=4, type=float,
        default=list(DEFAULT_PI),
        help="初始概率 π: S1 S2 S3 S4 (默认: 0.15 0.45 0.30 0.10)"
    )
    parser.add_argument(
        "--obs", nargs="+", type=str,
        default=["O1", "O2", "O3"],
        help="观测序列，用空格分隔 (默认: O1 O2 O3)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出每步递推过程"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="仅输出JSON格式（适合管道调用）"
    )

    args = parser.parse_args()

    pi = np.array(args.pi)
    obs_indices = parse_observations(args.obs)

    # 归一化pi
    pi = pi / pi.sum()

    final_dist, history = forward_algorithm(
        pi, DEFAULT_A, DEFAULT_B, obs_indices, verbose=args.verbose or args.json is False
    )

    if args.json:
        json_out = {
            "final_distribution": {
                "S1_无果结案": round(float(final_dist[0]), 4),
                "S2_和解罚款": round(float(final_dist[1]), 4),
                "S3_实质性处罚": round(float(final_dist[2]), 4),
                "S4_极端连锁": round(float(final_dist[3]), 4),
            },
            "most_likely": STATES[final_dist.argmax()],
            "suggested_position_pct": round(
                (1 - final_dist[2] - 2 * final_dist[3]) * 100, 1
            ),
        }
        print(json.dumps(json_out, ensure_ascii=False))
    else:
        print_result(final_dist, history)


if __name__ == "__main__":
    main()
