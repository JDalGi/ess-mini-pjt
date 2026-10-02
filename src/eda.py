"""Day 1의 배치별 탐색 통계와 그래프를 생성한다."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .preprocess import ROOT, FILES, GRID, load
from .features import knee

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig")
)
matplotlib.use("Agg")

OUT = ROOT / "results" / "eda"
FEATURES = [
    "delta_log_var",
    "delta_mean",
    "delta_min",
    "delta_std",
    "QD_mean",
    "QD_median",
    "QD_std",
    "QD_slope",
    "IR_mean",
    "IR_slope",
    "Tavg_mean",
    "Tmax_mean",
    "chargetime_mean",
    "rate1_C",
    "rate2_C",
    "switch_pct",
    "charge_I_mean",
    "charge_I_std",
    "charge_I_p95",
]

DISTRIBUTION_FEATURES = [
    "delta_log_var",
    "QD_mean",
    "QD_median",
    "QD_slope_10_100",
    "Tavg_mean",
    "IR_mean",
    "chargetime_mean",
    "charge_I_std_c10",
]


def feature_distribution_tables(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """핵심 입력의 분포와 Batch 1 관측 범위 밖 비율을 기술한다.

    셀 하나가 관측 단위로, 수명 관계를 해석하는 다른 표와 모집단을
    맞추기 위해 라벨 있는 셀을 사용하며, 결측은 대체하지 않고 센다.
    범위 밖 비율은 분포 이동의 기술 통계이며 셀 제거 기준은 아니다.
    """
    labeled = cells[cells.cycle_life.notna()]
    rows, shifts = [], []

    for label in FILES:
        group = labeled[labeled.batch.eq(label)]
        reference = labeled[labeled.batch.eq("Batch 1")]

        for feature in DISTRIBUTION_FEATURES:
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
            baseline = reference[feature].replace([np.inf, -np.inf], np.nan).dropna()
            q25, q75 = values.quantile([0.25, 0.75])
            rows.append(
                dict(
                    batch=label,
                    feature=feature,
                    count=len(values),
                    missing=len(group) - len(values),
                    zero_count=int(values.eq(0).sum()),
                    mean=values.mean(),
                    std=values.std(),
                    median=values.median(),
                    q25=q25,
                    q75=q75,
                    IQR=q75 - q25,
                    min=values.min(),
                    max=values.max(),
                    skew=values.skew(),
                )
            )
            # 같은 값 범위 안에서도 분포는 달라질 수 있어 중앙값 차이를 함께 기록한다.
            # 기준은 관측된 Batch 1의 값 범위다. 이 통계로 값을 제거하지 않는다.
            outside = (values < baseline.min()) | (values > baseline.max())
            shifts.append(
                dict(
                    batch=label,
                    feature=feature,
                    count=len(values),
                    median_difference_from_batch1=values.median() - baseline.median(),
                    outside_batch1_range_cells=int(outside.sum()),
                    outside_batch1_range_pct=100 * outside.mean()
                    if len(values)
                    else np.nan,
                )
            )

    return pd.DataFrame(rows), pd.DataFrame(shifts)


def feature_plan() -> pd.DataFrame:
    """EDA에 근거한 후보 비교 계획"""
    from .features import FEATURE_GROUPS
    reasons = {
        "A": "ΔQ 단일 피처 기준선",
        "B": "초기 용량 수준·강건 기울기의 추가 이득",
        "C": "대표 온도 하나의 추가 이득; IR·충전시간 보류",
        "D": "충전 정책 조합·부가 표기의 추가 이득",
        "E": "시간 가중 실측 전류 패턴의 추가 이득",
    }
    return pd.DataFrame([
        dict(stage=name, features=", ".join(columns), purpose=reasons[name])
        for name, columns in FEATURE_GROUPS.items()
    ])


def tables(cells, summary):
    labeled = cells[cells.cycle_life.notna()].copy()
    dist = []
    corr = []
    redundancy = []
    quality = []
    peers = []
    stage = []
    sensitivity = []
    vif = []

    for batch, g in cells.groupby("batch"):
        z = g[g.cycle_life.notna()]
        life = z.cycle_life
        q1, q3 = life.quantile([0.25, 0.75])
        dist.append(
            dict(
                batch=batch,
                total=len(g),
                labeled=len(z),
                missing=len(g) - len(z),
                min=life.min(),
                q1=q1,
                median=life.median(),
                mean=life.mean(),
                q3=q3,
                max=life.max(),
                std=life.std(),
                skew=life.skew(),
                short_count=int(life.lt(500).sum()),
                short_pct=100 * life.lt(500).mean(),
                long_count=int(life.gt(1000).sum()),
                long_pct=100 * life.gt(1000).mean(),
                low_fence=q1 - 1.5 * (q3 - q1),
                high_fence=q3 + 1.5 * (q3 - q1),
                low_outliers=int(life.lt(q1 - 1.5 * (q3 - q1)).sum()),
                high_outliers=int(life.gt(q3 + 1.5 * (q3 - q1)).sum()),
            )
        )

        for col in FEATURES:
            corr.append(
                dict(
                    batch=batch,
                    feature=col,
                    n=int(z[[col, "cycle_life"]].dropna().shape[0]),
                    spearman=z[col].corr(life, method="spearman"),
                    pearson=z[col].corr(life),
                )
            )

        for i, a in enumerate(FEATURES):
            for b in FEATURES[i + 1 :]:
                rho = z[a].corr(z[b], method="spearman")
                if abs(rho) >= 0.9:
                    redundancy.append(dict(batch=batch, a=a, b=b, spearman=rho))

        for r in z.nsmallest(3, "cycle_life").itertuples():
            p = z[z.policy.eq(r.policy) & z.cell_id.ne(r.cell_id)]
            peers.append(
                dict(
                    batch=batch,
                    cell_id=r.cell_id,
                    cycle_life=r.cycle_life,
                    policy=r.policy,
                    peer_n=len(p),
                    peer_life=p.cycle_life.median(),
                    Tavg=r.Tavg_mean,
                    peer_Tavg=p.Tavg_mean.median(),
                    delta_log_var=r.delta_log_var,
                    peer_delta_log_var=p.delta_log_var.median(),
                )
            )

        raw = summary[summary.batch.eq(batch)]

        quality.append(
            dict(
                batch=batch,
                placeholder_rows=int(raw.placeholder.sum()),
                endpoint_high_capacity_labeled=int(z.endpoint_high_capacity.sum()),
                min_early_rows=int(g.early_rows.min()),
                incomplete_delta=int(g.delta_valid_fraction.lt(1).sum()),
                IR_all_zero_cells=int(g.IR_zero_fraction.eq(1).sum()),
                unparsed_policy=int(g.rate1_C.isna().sum()),
                QD_over_1_3=int(raw.QD.gt(1.3).sum()),
                nonfinite_QD=int((~np.isfinite(raw.QD)).sum()),
            )
        )

        stage.append(
            dict(
                batch=batch,
                cells=len(z),
                accelerated=int(z.accelerates.sum()),
                acceleration_pct=100 * z.accelerates.mean(),
                knee_n=int(z.knee_cycle.notna().sum()),
                knee_median=z.knee_cycle.median(),
                slope_mid_median=z.slope_mid.median(),
                slope_late_median=z.slope_late.median(),
                current_vs_early=z.charge_I_p95.corr(z.QD_slope, method="spearman"),
                current_vs_late=z.charge_I_p95.corr(-z.slope_late, method="spearman"),
            )
        )

        eligible = z[~z.endpoint_high_capacity]

        sensitivity.append(
            dict(
                batch=batch,
                n_raw=len(z),
                n_endpoint_screened=len(eligible),
                delta_raw=z.delta_log_var.corr(life, method="spearman"),
                delta_screened=eligible.delta_log_var.corr(
                    eligible.cycle_life, method="spearman"
                ),
                ols_rho=z.QD_slope_ols.corr(life, method="spearman"),
                robust_rho=z.QD_slope.corr(life, method="spearman"),
            )
        )

        cols = [
            "delta_log_var",
            "QD_slope",
            "Tavg_mean",
            "Tmax_mean",
            "charge_I_p95",
            "rate1_C",
            "switch_pct",
        ]
        data = z[cols].dropna()
        active = [c for c in cols if data[c].std() > 1e-12]
        x = ((data[active] - data[active].mean()) / data[active].std()).to_numpy()

        for j, col in enumerate(active):
            other = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
            residual = x[:, j] - other @ np.linalg.lstsq(other, x[:, j], rcond=None)[0]
            frac = np.square(residual).sum() / np.square(x[:, j]).sum()
            vif.append(
                dict(
                    batch=batch,
                    feature=col,
                    n=len(x),
                    VIF=1 / frac if frac > 1e-12 else np.inf,
                )
            )

    distributions = []

    for (batch, col), v in labeled.melt(
        id_vars=["batch", "cell_id"], value_vars=FEATURES
    ).groupby(["batch", "variable"]):
        x = v.value.dropna()
        a, b = x.quantile([0.25, 0.75])
        distributions.append(
            dict(
                batch=batch,
                feature=col,
                n=len(x),
                missing=len(v) - len(x),
                mean=x.mean(),
                std=x.std(),
                median=x.median(),
                q1=a,
                q3=b,
                min=x.min(),
                max=x.max(),
            )
        )

    policies = (
        labeled.groupby(["batch", "policy"], dropna=False)
        .agg(
            n=("cell_id", "size"),
            mean_life=("cycle_life", "mean"),
            std_life=("cycle_life", "std"),
            median_life=("cycle_life", "median"),
        )
        .reset_index()
    )

    groups = (
        labeled.assign(
            life_group=np.select(
                [labeled.cycle_life.lt(500), labeled.cycle_life.gt(1000)],
                ["short <500", "long >1000"],
                default="middle",
            )
        )
        .groupby(["batch", "life_group"])
        .agg(
            n=("cell_id", "size"),
            delta_log_var=("delta_log_var", "median"),
            Tavg=("Tavg_mean", "median"),
            rate1=("rate1_C", "median"),
        )
        .reset_index()
    )

    tbl = dict(
        cells=cells,
        distribution=pd.DataFrame(dist),
        quality=pd.DataFrame(quality),
        correlations=pd.DataFrame(corr),
        redundancy=pd.DataFrame(redundancy),
        short_cell_peers=pd.DataFrame(peers),
        degradation=pd.DataFrame(stage),
        sensitivity=pd.DataFrame(sensitivity),
        vif=pd.DataFrame(vif),
        feature_distributions=pd.DataFrame(distributions),
        policies=policies,
        life_groups=groups,
        endpoint_flags=cells[cells.cycle_life.notna() & cells.endpoint_high_capacity],
        label_audit=cells[
            [
                "batch",
                "cell_id",
                "cycle_life",
                "observed_cycles",
                "label_minus_record_end",
                "QD_final",
                "QD_at_label",
                "first_QD_le_088",
                "endpoint_high_capacity",
            ]
        ],
        missing_labels=cells[cells.cycle_life.isna()],
    )

    return tbl


def delta_group_comparison(cells):
    """보고서 및 Delta-Q 그림과 동일한 수명 그룹"""
    rows = []
    labeled = cells[cells.cycle_life.notna()]

    for batch, group in labeled.groupby("batch"):
        if batch == "Batch 2":
            groups = [
                ("단수명 <500회", group[group.cycle_life.lt(500)]),
                ("장수명 >1000회", group[group.cycle_life.gt(1000)]),
            ]
        else:
            q1, q3 = group.cycle_life.quantile([0.25, 0.75])
            groups = [
                ("수명 하위 사분위", group[group.cycle_life.le(q1)]),
                ("수명 상위 사분위", group[group.cycle_life.ge(q3)]),
            ]
        for name, subset in groups:
            rows.append(
                dict(
                    batch=batch,
                    group=name,
                    n=len(subset),
                    delta_min_median=subset.delta_min.median(),
                    delta_log_var_median=subset.delta_log_var.median(),
                )
            )

    return pd.DataFrame(rows)


def figures(cells, summary, curves, traces):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    labels = list(FILES)
    labeled = cells[cells.cycle_life.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    for ax, batch in zip(axes, labels):
        x = labeled[labeled.batch.eq(batch)].cycle_life
        ax.hist(
            x,
            bins=np.r_[np.arange(150, 2300, 100), 2300],
            color="#4477aa",
            edgecolor="white",
        )
        ax.axvline(500, ls=":", color="#cc3311")
        ax.axvline(1000, ls=":", color="#0077bb")
        ax.axvline(x.median(), ls="--", color="black", label=f"median {x.median():.0f}")
        ax.set(
            title=f"{batch}: n={len(x)}",
            xlabel="Recorded cycle life",
            ylabel="Cells",
            xlim=(150, 2300),
        )
        ax.legend()

    save(fig, "01_life")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    legend = [
        Line2D([0], [0], color=c, label=t)
        for c, t in [
            ("#cc3311", "life <500"),
            ("#0077bb", "life >1000"),
            ("#999999", "middle / unlabeled"),
        ]
    ]

    for j, batch in enumerate(labels):
        for r in cells[cells.batch.eq(batch)].itertuples():
            g = summary[summary.cell_id.eq(r.cell_id) & ~summary.placeholder]
            if np.isfinite(r.cycle_life):
                g = g[g.cycle.le(r.cycle_life)]
            color = (
                "#cc3311"
                if r.cycle_life < 500
                else "#0077bb"
                if r.cycle_life > 1000
                else "#999999"
            )
            for ax in axes[:, j]:
                ax.plot(g.cycle, g.QD, color=color, lw=0.6, alpha=0.65)
        axes[0, j].set(title=f"{batch}: QD up to recorded life", ylabel="QD (Ah)")
        axes[1, j].set(
            title=f"{batch}: zoom (values retained)",
            xlabel="Cycle",
            ylabel="QD (Ah)",
            ylim=(0.75, 1.18),
        )
        axes[1, j].axhline(0.88, color="black", ls=":", label="0.88 Ah reference")
        axes[0, j].legend(handles=legend, fontsize=8)
    save(fig, "02_degradation")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, batch in zip(axes, labels):
        g = labeled[labeled.batch.eq(batch) & labeled.knee_cycle.notna()]
        r = g.iloc[np.argmin(abs(g.cycle_life - g.cycle_life.median()))]
        s = summary[
            summary.cell_id.eq(r.cell_id)
            & ~summary.placeholder
            & summary.cycle.le(r.cycle_life)
        ]
        ax.plot(s.cycle, s.QD, lw=0.6, color="#aaaaaa", label="raw")
        ax.plot(
            s.cycle,
            s.QD.rolling(21, center=True, min_periods=11).median(),
            color="#4477aa",
            label="median 21",
        )
        ax.axvline(
            r.knee_cycle,
            color="#cc3311",
            ls="--",
            label=f"knee candidate {r.knee_cycle:.0f}",
        )
        ax.set(
            title=f"{batch}: {r.cell_id}",
            xlabel="Cycle",
            ylabel="QD (Ah)",
            ylim=(0.8, 1.15),
        )
        ax.legend(fontsize=8)
    save(fig, "03_knee")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, batch in zip(axes, labels):
        g = labeled[labeled.batch.eq(batch)]
        if g.cycle_life.lt(500).any() and g.cycle_life.gt(1000).any():
            subsets = [
                (g[g.cycle_life.lt(500)], "#cc3311", "short <500"),
                (g[g.cycle_life.gt(1000)], "#0077bb", "long >1000"),
            ]
        else:
            q1, q3 = g.cycle_life.quantile([0.25, 0.75])
            subsets = [
                (g[g.cycle_life.le(q1)], "#cc3311", "bottom quartile"),
                (g[g.cycle_life.ge(q3)], "#0077bb", "top quartile"),
            ]
        for sub, color, title in subsets:
            y = np.stack([curves[c] for c in sub.cell_id])
            med = np.nanmedian(y, axis=0)
            ax.plot(GRID, med, color=color, label=f"{title} n={len(sub)}")
            a, b = np.nanquantile(y, [0.25, 0.75], axis=0)
            ax.fill_between(GRID, a, b, color=color, alpha=0.15)
        ax.axhline(0, color="grey", ls=":")
        ax.set(title=batch, xlabel="Voltage (V)", ylabel="Q100(V) - Q10(V) (Ah)")
        ax.legend(fontsize=8)
    save(fig, "04_delta_q")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, batch in zip(axes, labels):
        g = labeled[labeled.batch.eq(batch)]
        ax.scatter(g.delta_log_var, g.cycle_life, color="#4477aa")
        ax.set(
            title=f"{batch}: rho={g.delta_log_var.corr(g.cycle_life, method='spearman'):.3f}",
            xlabel="log10(var(delta Q))",
            ylabel="Recorded cycle life",
        )
    save(fig, "05_delta_life")

    for batch in labels:
        g = (
            labeled[labeled.batch.eq(batch)]
            .groupby("policy")
            .cycle_life.agg(["mean", "std", "count"])
            .sort_values("mean")
        )
        fig, ax = plt.subplots(figsize=(11, max(5, len(g) * 0.27)))
        ax.barh(
            np.arange(len(g)),
            g["mean"],
            xerr=g["std"].fillna(0),
            color="#4477aa",
            alpha=0.8,
        )
        ax.set_yticks(
            np.arange(len(g)),
            [f"{p} (n={int(n)})" for p, n in zip(g.index, g["count"])],
            fontsize=8,
        )
        ax.set(
            title=f"{batch}: policy mean life +/- sample SD (n=1: SD unavailable)",
            xlabel="Recorded cycle life",
        )
        save(fig, f"06_policy_b{batch[-1]}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for j, batch in enumerate(labels):
        g = labeled[labeled.batch.eq(batch)]
        p = axes[0, j].scatter(
            g.rate1_C, g.cycle_life, c=g.switch_pct, cmap="viridis", s=32
        )
        fig.colorbar(p, ax=axes[0, j], label="Switch SOC (%)")
        axes[0, j].set(
            title=batch, xlabel="First-stage C-rate", ylabel="Recorded cycle life"
        )
        axes[1, j].scatter(g.charge_I_p95, -g.slope_late, color="#4477aa")
        axes[1, j].set(
            xlabel="Cycle 10 time-weighted current p95 (A)",
            ylabel="Late QD loss rate (Ah/cycle)",
        )
    save(fig, "07_current")
    chosen = [
        "delta_log_var",
        "QD_slope",
        "Tavg_mean",
        "Tmax_mean",
        "chargetime_mean",
        "IR_mean",
        "charge_I_std",
        "rate1_C",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, batch in zip(axes, labels):
        g = labeled[labeled.batch.eq(batch)]
        m = g[chosen + ["cycle_life"]].corr(method="spearman")
        im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(m)), m.columns, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(m)), m.index, fontsize=8)
        ax.set_title(batch)
        for i in range(len(m)):
            for j in range(len(m)):
                v = m.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(v) > 0.6 else "black",
                )
    fig.subplots_adjust(bottom=0.30, right=0.91, wspace=0.55)
    color_axis = fig.add_axes([0.935, 0.30, 0.012, 0.55])
    fig.colorbar(im, cax=color_axis, label="Spearman rho")
    fig.savefig(OUT / "figures" / "08_correlations.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for ax, col in zip(
        axes.ravel(),
        [
            "delta_log_var",
            "QD_median",
            "QD_slope",
            "Tavg_mean",
            "chargetime_mean",
            "IR_mean",
        ],
    ):
        ax.boxplot(
            [labeled[labeled.batch.eq(b)][col].dropna() for b in labels],
            tick_labels=labels,
            showfliers=True,
        )
        ax.set(title=col, ylabel="Feature value")
    save(fig, "09_feature_distributions")


def run_day1():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "eda").mkdir(parents=True, exist_ok=True)
    cells, summary, curves, traces = load()
    assert cells.cell_id.is_unique and len(cells) == 139
    assert cells.early_rows.eq(99).all()
    assert cells.delta_valid_fraction.eq(1).all()
    tbl = tables(cells, summary)
    # 평활화 창 크기에 따라 Knee 후보가 얼마나 달라지는지 확인한다.
    ks = []
    for window in [11, 21, 41]:
        for batch, g in cells[cells.cycle_life.notna()].groupby("batch"):
            detected = []
            for r in g.itertuples():
                s = summary[
                    summary.cell_id.eq(r.cell_id)
                    & ~summary.placeholder
                    & summary.cycle.le(r.cycle_life)
                ]
                detected.append(
                    knee(s.cycle.to_numpy(), s.QD.to_numpy(), window)["knee_cycle"]
                )
            ks.append(
                dict(
                    batch=batch,
                    window=window,
                    count=int(np.isfinite(detected).sum()),
                    median=float(np.nanmedian(detected)),
                )
            )
    tbl["knee_sensitivity"] = pd.DataFrame(ks)
    for name, frame in tbl.items():
        frame.to_csv(ROOT / "results" / "eda" / f"{name}.csv", index=False)
    np.savez_compressed(
        ROOT / "results" / "eda" / "delta_curves.npz", voltage=GRID, **curves
    )
    figures(cells, summary, curves, traces)

    audit = dict(
        input_files=FILES,
        raw_cells=len(cells),
        labeled_cells=int(cells.cycle_life.notna().sum()),
        early_complete_cells=int(cells.early_rows.eq(99).sum()),
        delta_complete_cells=int(cells.delta_valid_fraction.eq(1).sum()),
        summary_rows=len(summary),
        analysis_unit="one battery cell",
        sources=["notebooks/00_Scratch.ipynb", "data/*.mat"],
        ignored_directories=["report", "src_initial", "tests"],
    )
    (OUT / "analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )

    for name in [
        "distribution",
        "quality",
        "degradation",
        "sensitivity",
        "life_groups",
        "short_cell_peers",
        "knee_sensitivity",
    ]:
        print("\n" + name + "\n" + tbl[name].to_string(index=False), flush=True)

    print(
        "\nTOP CORRELATIONS\n"
        + tbl["correlations"]
        .assign(abs_rho=lambda d: d.spearman.abs())
        .sort_values("abs_rho", ascending=False)
        .groupby("batch")
        .head(5)
        .to_string(index=False),
        flush=True,
    )

    return cells, summary, curves, traces, tbl


if __name__ == "__main__":
    run_day1()
