# Antitrust Investigation HMM Framework 🎲

**反垄断调查 HMM 概率推演框架**

A Hidden Markov Model (HMM) framework for event-driven antitrust investigation analysis. Supports multi-company competitive landscape analysis, stock price behavior modeling, and probabilistic scenario inference via the Forward Algorithm.

> **Language note:** This README is primarily in English, with Chinese annotations for key concepts. The SKILL.md and Python docstrings contain full Chinese documentation.  
> 本 README 以英文为主，关键概念附中文注释。完整中文文档见 SKILL.md 和 Python 代码注释。

---

## Overview | 概览

When a company (or its competitors) faces an antitrust/price-fixing investigation, this framework provides systematic analysis across 7 layers:

1. **Event Timeline & Research** — Investigation scope, involved parties, legal framework
2. **Multi-Company Competitive Analysis** — Market share, customer concentration, financial comparison
3. **Stock Price Behavior** — Event shock measurement, volume-price patterns, relative strength
4. **HMM Probabilistic Inference** ⭐ — Core forward algorithm probability engine
5. **Investment Decision Matrix** — Scenario-based pricing & position sizing
6. **Cross-Company Differential Analysis** — Who gets hit hardest, who is oversold?
7. **Structured Conclusion** — FACT / OPINION / INFERENCE annotated output

### States | 状态定义

| State | Description | Chinese |
|:-----:|:------------|:--------|
| S1 | Case dismissed / minor penalty | 无果结案 |
| S2 | Settlement with fine | 和解罚款 |
| S3 | Substantial penalty | 实质性处罚 |
| S4 | Extreme chain reaction | 极端连锁反应 |

### Observations | 观测变量

| Obs | Event | Chinese |
|:---:|:------|:--------|
| O1 | No new developments, company "cooperating" | 检方无新进展 |
| O2 | Preliminary findings announced | 初步调查结论 |
| O3 | Indictment / executive prosecution | 移交审判/起诉高管 |
| O4 | Large fine provision announced | 大额罚款计提 |
| O5 | Customers initiate alternate supplier validation | 客户启动替代供应商验证 |
| O6 | US DOJ announces findings/fines | 美国DOJ宣布结果 |
| O7 | Customers announce price cuts/compensation | 客户发布降价/赔偿公告 |

---

## Quick Start | 快速开始

### Prerequisites | 前置条件

- Python 3.8+
- NumPy

```bash
pip install numpy
```

### Run with Default Parameters | 默认参数运行

```bash
python hmm_forward.py
```

Output (default: Montage Technology / 澜起科技 case, 3-step inference):

```
最终概率分布（共 3 步递推）:
  S1:无果结案                  15.3%
  S2:和解罚款                  47.3%  ← most likely
  S3:实质性处罚                 32.1%
  S4:极端连锁                   5.3%

仓位建议: 57.2% 仓位
```

### Customize Initial Probability | 自定义初始概率

```bash
python hmm_forward.py --pi 0.1 0.3 0.4 0.2 --obs O1 O2 O3 --verbose
```

### Custom Observation Sequence | 自定义观测序列

```bash
python hmm_forward.py --obs O1 O2 O3 O4 O5
```

### JSON Output | JSON 输出（管道友好）

```bash
python hmm_forward.py --pi 0.05 0.25 0.45 0.25 --obs O1 O2 O3 O4 --json
```

```json
{
  "final_distribution": {
    "S1_无果结案": 0.1234,
    "S2_和解罚款": 0.4567,
    "S3_实质性处罚": 0.3456,
    "S4_极端连锁": 0.0743
  },
  "most_likely": "S2:和解罚款",
  "suggested_position_pct": 50.6
}
```

---

## Parameters | 参数说明

### Transition Matrix A | 状态转移矩阵

```
        S1     S2     S3     S4
S1  [  0.95,  0.04,  0.01,  0.00 ]
S2  [  0.05,  0.85,  0.08,  0.02 ]
S3  [  0.00,  0.10,  0.80,  0.10 ]
S4  [  0.00,  0.00,  0.15,  0.85 ]
```

High diagonal values reflect state inertia — investigations follow path-dependent processes.

### Observation Matrix B | 观测矩阵

```
        O1    O2    O3    O4    O5    O6    O7
S1  [  0.50, 0.20, 0.02, 0.01, 0.01, 0.10, 0.16 ]
S2  [  0.15, 0.35, 0.05, 0.05, 0.05, 0.10, 0.25 ]
S3  [  0.02, 0.15, 0.20, 0.15, 0.15, 0.10, 0.23 ]
S4  [  0.01, 0.05, 0.15, 0.10, 0.20, 0.20, 0.29 ]
```

### Position Sizing Formula | 仓位计算公式

```
position = base_position × (1 - P(S3) - 2 × P(S4))
```

---

## Project Structure | 项目结构

```
antitrust-hmm-investigation/
├── README.md           # This file (English + Chinese)
├── SKILL.md            # Full framework documentation (Chinese)
└── hmm_forward.py      # HMM Forward Algorithm implementation
```

---

## Algorithm | 算法

The framework uses the **Forward Algorithm** for HMM inference:

1. **Initialize** t=1: α₁(Sj) = πj × Bj(O₁), then normalize
2. **Recur** t>1: αₜ(Sj) = [Σαₜ₋₁(Si) × A_ij] × Bj(Oₜ), then normalize
3. **Result**: Final normalized α_T gives P(State | Observations)

The `--verbose` flag shows each step's prediction and posterior distribution.

---

## Use Case Example | 案例参考

This framework was developed during the **July 2026 South Korean antitrust raids** on three memory interface chip companies: Montage Technology (澜起科技), Renesas, and Rambus. The full analysis covered:

- Market structure: Montage at ~36.8% global share, three firms combined >93%
- Customer concentration: Samsung, SK Hynix, Micron as common customers
- Price action: Montage dropped 16.44% on event day with 255.8B RMB volume
- HMM inference: Most likely path → settlement with fine (S2)

---

## Data Discipline | 数据纪律

- All financial data defaults to **Wind MCP** (万得) as primary source
- Web search for qualitative info only (news, announcements, industry trends)
- Every data point tagged with source and date
- PE ratios strictly annotated: PE(LYR) / PE(TTM) / PE(预测) / PE(远期)
- Output annotated as: [FACT] / [OPINION] / [INFERENCE] / [TRIANGULATED_FACT]

---

## License | 许可

MIT

---

## Related | 相关

- [Serenity Chokepoint Investing Framework](https://github.com/yuyang-rgb094/serenity-chokepoint-investing-enhanced) — AI infrastructure supply-chain bottleneck analysis
- Full framework documentation (Chinese): [`SKILL.md`](./SKILL.md)
