"""100번째 사이클 종료 시점까지 관측할 수 있는 피처를 생성한다."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from src.preprocess import VOLTAGE_GRID


SUMMARY_SIGNALS = ["QD", "QC", "IR", "Tmax", "Tavg", "Tmin", "chargetime"]


def placeholder_rows(summary: pd.DataFrame) -> pd.Series:
    """Batch 1의 첫 사이클처럼 요약값이 모두 0이고 시계열이 빈 기록을 구분한다."""
    if "is_placeholder" in summary:
        return summary.is_placeholder.fillna(False).astype(bool)
    if "placeholder" in summary:
        return summary.placeholder.fillna(False).astype(bool)
    if not set(SUMMARY_SIGNALS).issubset(summary.columns):
        return pd.Series(False, index=summary.index)
    return summary[SUMMARY_SIGNALS].eq(0).all(axis=1) & summary.cycle.eq(1)


def flag_qd(summary: pd.DataFrame) -> pd.Series:
    """계산할 수 없는 값과 빈 기록, 음의 누적 방전 용량만 표시한다.

    유한한 양의 용량은 크거나 작다는 이유로 삭제하지 않는다.
    """
    return ~np.isfinite(summary.QD) | summary.QD.lt(0) | placeholder_rows(summary)


def sanitize_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """원본을 보존하고 계산 불가 값만 NaN으로 구분한다.

    큰 양의 급증은 측정 오류인지 특수 실험인지 확정할 수 없으므로 유지한다.
    평균·표준편차와 중앙값·강건 기울기를 함께 계산해 그 영향을 비교한다.
    """
    clean = summary.copy()
    zero = placeholder_rows(summary)
    clean[SUMMARY_SIGNALS] = clean[SUMMARY_SIGNALS].replace([np.inf, -np.inf], np.nan)
    clean.loc[zero, SUMMARY_SIGNALS] = np.nan
    clean.loc[flag_qd(summary), "QD"] = np.nan
    return clean


def capacity_slope(group: pd.DataFrame, robust: bool = True) -> float:
    """QD 변화량(Ah/cycle). 양수는 용량 증가이며 열화 속도로 단정하지 않는다.

    Theil–Sen은 점 쌍 기울기의 중앙값이라 단발성 급증의 영향을 줄인다.
    OLS도 보존해 계산 방식에 대한 민감도를 확인한다. 어느 방법을 쓸지는
    이후 Batch 1 검증에서 선택하며, 두 기울기를 항상 함께 넣지는 않는다.
    """
    valid = group.loc[np.isfinite(group.cycle) & ~flag_qd(group)].sort_values("cycle")
    if valid.cycle.nunique() < 2:
        return np.nan  # 기울기 계산에는 서로 다른 사이클 값이 두 개 이상 필요하다.
    if robust:
        return float(theilslopes(valid.QD, valid.cycle).slope)
    return float(np.polyfit(valid.cycle, valid.QD, 1)[0])


def charge_statistics(trace: dict) -> dict[str, float]:
    """양의 충전 전류 구간을 시간 가중한다. 방전·휴지 구간을 연결하지 않는다."""
    # 측정 지점 평균은 촘촘하게 기록한 구간을 과대 반영한다.
    # 구간 시간으로 가중하고, 전류가 선형으로 변한다는 근사로 모멘트를 적분한다.
    # p95는 시간 가중 구간 중간 전류의 백분위이며 정확한 연속 분포 추정은 아니다.
    current, time = np.asarray(trace["I"]), np.asarray(trace["t"])
    if current.ndim != 1 or time.ndim != 1 or len(current) != len(time):
        raise ValueError("Current and time must be aligned 1D arrays")
    dt = np.diff(time)
    valid = (
        np.isfinite(current[:-1])
        & np.isfinite(current[1:])
        & np.isfinite(dt)
        & (dt > 0)
        & (current[:-1] > 0)
        & (current[1:] > 0)
    )
    weights = dt[valid]
    diagnostic = dict(
        time_reversals=int(np.sum(dt < 0)), invalid_intervals=int(np.sum(~valid))
    )
    values = (current[:-1][valid] + current[1:][valid]) / 2
    if not len(values):
        return dict(mean=np.nan, std=np.nan, p95=np.nan, **diagnostic)
    mean = float(np.average(values, weights=weights))
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    # 측정점 사이 전류가 선형으로 변한다고 가정한 2차 모멘트다.
    a, b = current[:-1][valid], current[1:][valid]
    second = np.average((a * a + a * b + b * b) / 3, weights=weights)
    return dict(
        mean=mean,
        std=float(np.sqrt(max(0.0, second - mean * mean))),
        p95=float(values[order][np.searchsorted(cumulative, 0.95)]),
        **diagnostic,
    )


def policy_rates(policy: str) -> tuple[float, float, float]:
    """정책명을 첫 속도·전환율·둘째 속도로 분해해 정책 조합을 표현한다.

    newstructure 같은 부가 표기는 따로 보존하며, VarCharge 형식은
    일반 두 단계 정책으로 임의 해석하지 않고 결측으로 남긴다.
    """
    match = re.match(r"^([\d.]+)C\(([\d.]+)%\)-([\d.]+)C(?:$|[-(])", policy)
    if not match:
        return np.nan, np.nan, np.nan
    values = tuple(map(float, match.groups()))
    return (
        values
        if values[0] > 0 and values[2] > 0 and 0 <= values[1] <= 100
        else (np.nan,) * 3
    )


def add_early_features(
    cells: pd.DataFrame, summary: pd.DataFrame, mask_qd: bool = True
) -> pd.DataFrame:
    """배터리 한 개를 한 행으로 요약한다. 사이클을 독립 표본으로 늘리지 않는다.

    cycle 2~100은 배치 공통 관측 구간이다. 완전성·유효 개수는 기록하되
    임의 개수 기준으로 피처를 삭제하지 않는다. 결측 대체와 표준화는
    이 함수에서 수행하지 않고 모델 학습 폴드 안에서만 적합한다.
    """
    # Batch 1의 첫 사이클은 빈 기록이다. 모든 배치에 2~100사이클을 적용한다.
    early = summary.loc[summary.cycle.between(2, 100)].copy()
    early["QD_flagged"] = flag_qd(early)
    if mask_qd:
        early = sanitize_summary(early)
    grouped = early.groupby("cell_id")
    features = grouped.agg(
        early_rows=("cycle", "size"),
        QD_valid_rows=("QD", "count"),
        QD_flagged_rows=("QD_flagged", "sum"),
        QD_mean=("QD", "mean"),
        QD_std=("QD", "std"),
        QD_median=("QD", "median"),
        IR_mean=("IR", "mean"),
        IR_std=("IR", "std"),
        Tavg_mean=("Tavg", "mean"),
        Tmax_mean=("Tmax", "mean"),
        chargetime_mean=("chargetime", "mean"),
    )

    features["early_complete"] = grouped.cycle.apply(
        lambda x: set(x) == set(range(2, 101)) and len(x) == 99
    )
    features["QD_valid_fraction"] = features.QD_valid_rows / 99
    for robust, name in [(True, "QD_slope_10_100"), (False, "QD_slope_10_100_ols")]:
        features[name] = grouped.apply(
            lambda g: capacity_slope(g[g.cycle.between(10, 100)], robust=robust),
            include_groups=False,
        )
    return cells.merge(
        features.reset_index(), on="cell_id", how="left", validate="one_to_one"
    )


def build_features(
    cells: pd.DataFrame,
    summary: pd.DataFrame,
    curves: dict[str, np.ndarray],
    current_traces: dict[str, dict],
) -> pd.DataFrame:
    """정책·사이클 요약·ΔQ 피처를 결합해 셀당 한 행으로 정리한다."""
    cells = add_early_features(cells, summary)
    rates = cells.policy.apply(policy_rates)
    cells[["rate1_C", "switch_pct", "rate2_C"]] = pd.DataFrame(
        rates.tolist(), index=cells.index
    )
    cells["new_structure"] = cells.policy.str.contains("newstructure", case=False)
    # 구조 표기는 배치 차이 해석용이다. Batch 1에서는 모두 False이므로
    # Batch 1만으로 구조 효과를 학습할 수 없고 후보 입력 목록에도 넣지 않는다.

    rows = []
    for cell_id, delta in curves.items():
        finite = delta[np.isfinite(delta)]
        # 일부 전압 구간만 사용하면 통계량의 비교 범위가 달라진다.
        complete = len(finite) == VOLTAGE_GRID.size
        variance = float(np.var(finite)) if complete else np.nan
        charge10 = current_traces[cell_id][10]["I"]
        charge100 = current_traces[cell_id][100]["I"]
        charge10 = charge10[np.isfinite(charge10) & (charge10 > 0)]
        charge100 = charge100[np.isfinite(charge100) & (charge100 > 0)]
        time10 = charge_statistics(current_traces[cell_id][10])
        time100 = charge_statistics(current_traces[cell_id][100])
        rows.append(
            {
                "cell_id": cell_id,
                "delta_valid_fraction": len(finite) / VOLTAGE_GRID.size,
                "delta_mean": float(np.mean(finite)) if complete else np.nan,
                "delta_min": float(np.min(finite)) if complete else np.nan,
                "delta_var": variance,
                "delta_log_var": np.log10(variance) if variance > 0 else np.nan,
                "charge_I_sample_mean_c10": float(np.mean(charge10))
                if len(charge10)
                else np.nan,
                "charge_I_sample_std_c10": float(np.std(charge10))
                if len(charge10)
                else np.nan,
                "charge_I_sample_p95_c10": float(np.percentile(charge10, 95))
                if len(charge10)
                else np.nan,
                "charge_I_mean_c10": time10["mean"],
                "charge_I_std_c10": time10["std"],
                "charge_I_p95_c10": time10["p95"],
                "charge_I_p95_c100": time100["p95"],
                "charge_time_reversals_c10": time10["time_reversals"],
                "charge_time_reversals_c100": time100["time_reversals"],
            }
        )
    return cells.merge(
        pd.DataFrame(rows), on="cell_id", how="left", validate="one_to_one"
    )


def slope(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    return float(theilslopes(y[valid], x[valid]).slope) if valid.sum() >= 3 else np.nan


def current_stats(t, i):
    if len(t) != len(i) or len(t) < 2:
        return dict(charge_I_mean=np.nan, charge_I_std=np.nan, charge_I_p95=np.nan)
    dt = np.diff(t)
    valid = (
        np.isfinite(dt)
        & (dt > 0)
        & np.isfinite(i[:-1])
        & np.isfinite(i[1:])
        & (i[:-1] > 0)
        & (i[1:] > 0)
    )
    a, b, w = i[:-1][valid], i[1:][valid], dt[valid]
    if not len(w):
        return dict(charge_I_mean=np.nan, charge_I_std=np.nan, charge_I_p95=np.nan)
    mid = (a + b) / 2
    mean = np.average(mid, weights=w)
    second = np.average((a * a + a * b + b * b) / 3, weights=w)
    order = np.argsort(mid)
    p95 = mid[order][np.searchsorted(np.cumsum(w[order]) / w.sum(), 0.95)]
    return dict(
        charge_I_mean=float(mean),
        charge_I_std=float(np.sqrt(max(0, second - mean * mean))),
        charge_I_p95=float(p95),
    )


def knee(x, y, window=21):
    """이동 중앙값으로 평활화한 곡선에 연속 두 구간 직선을 적합해 Knee 후보를 찾는다."""
    good = np.isfinite(x) & np.isfinite(y) & (y > 0) & (x >= 10)
    x, y = x[good], y[good]
    empty = dict(
        knee_cycle=np.nan, knee_before=np.nan, knee_after=np.nan, knee_gain=np.nan
    )
    if len(x) < 80:
        return empty
    y = (
        pd.Series(y)
        .rolling(window, center=True, min_periods=(window + 1) // 2)
        .median()
        .to_numpy()
    )
    x, y = x[::5], y[::5]
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    base = np.column_stack([np.ones(len(x)), x])
    base_err = np.square(y - base @ np.linalg.lstsq(base, y, rcond=None)[0]).sum()
    best = None
    # 두 구간 모두 감소하고, 후기 감소가 중기보다 2배 이상 빠른 후보만 검토한다.
    for k in x[int(0.2 * len(x)) : int(0.85 * len(x))]:
        design = np.column_stack([base, np.maximum(0, x - k)])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        pre, post = beta[1], beta[1] + beta[2]
        err = np.square(y - design @ beta).sum()
        if pre < 0 and post < 2 * pre and (best is None or err < best[0]):
            best = (err, k, pre, post)
    if best is not None and base_err > 0 and best[0] < 0.8 * base_err:
        return dict(
            knee_cycle=float(best[1]),
            knee_before=float(best[2]),
            knee_after=float(best[3]),
            knee_gain=float(1 - best[0] / base_err),
        )
    return empty


FEATURE_GROUPS = {
    "A": ["delta_log_var"],
    "B": ["delta_log_var", "QD_median", "QD_slope"],
    "C": ["delta_log_var", "Tavg_mean"],
    "D": ["delta_log_var", "rate1_C", "switch_pct", "rate2_C", "newstructure"],
    "E": ["delta_log_var", "charge_I_mean", "charge_I_std", "charge_I_p95"],
}

# 보고서의 A~E를 유지하고, 추가 추출·대표 변수 교체 후보를 별도로 표시한다.
SEARCH_GROUPS = {
    **FEATURE_GROUPS,
    "B_QD_change": FEATURE_GROUPS["B"] + ["QD_change"],
    "B_QD_relative": FEATURE_GROUPS["B"] + ["QD_relative_change"],
    "C_Tmax": ["delta_log_var", "Tmax_mean"],
    "D_interaction": FEATURE_GROUPS["D"] + ["policy_interaction"],
    "A_delta_min": ["delta_min"],
    "A_delta_mean": ["delta_mean"],
}
GROUP_PARENTS = {
    "B": "A", "C": "A", "D": "A", "E": "A",
    "B_QD_change": "B", "B_QD_relative": "B",
    "C_Tmax": "A", "D_interaction": "D",
}
FORBIDDEN_FEATURES = {
    "cycle_life", "cell_id", "batch", "observed_cycles", "n_cycles",
    "QD_final", "QD_at_label", "first_QD_le_088", "knee_cycle",
    "knee_before", "knee_after", "knee_gain", "slope_mid", "slope_late",
    "accelerates", "analysis_last_cycle", "endpoint_high_capacity",
    "end_relative_capacity", "label_minus_record_end",
}


def feature_manifest():
    """Day 1 근거·관측 기간과 학습 입력의 대응을 저장한다."""
    rows = []
    for name in dict.fromkeys(c for cols in SEARCH_GROUPS.values() for c in cols):
        if name.startswith("delta_"):
            source, cycles, rationale = "Qdlin / Vdlin", "10, 100", "전압별 용량 변화의 대표 통계"
        elif name in {"rate1_C", "rate2_C", "switch_pct", "newstructure", "policy_interaction"}:
            source, cycles, rationale = "policy_readable", "실험 시작 시", "충전 단계·전환 조건·부가 표기"
        elif name.startswith("charge_I_"):
            source, cycles, rationale = "I / t", "10", "시간 가중 실측 충전 전류"
        else:
            source = "summary"
            cycles = "10, 100" if "change" in name else "10–100" if "slope" in name else "2–100"
            rationale = "초기 용량 수준·변화 또는 대표 온도"
        rows.append(dict(feature=name, source=source, cycles=cycles, rationale=rationale))
    return pd.DataFrame(rows)


def early_features(early, change, delta, t, current):
    """100사이클까지 관측한 정보만 초기 피처로 추출한다."""
    full = np.isfinite(delta).all()
    var = np.var(delta) if full else np.nan
    return dict(
        delta_mean=float(np.mean(delta)) if full else np.nan,
        delta_min=float(np.min(delta)) if full else np.nan,
        delta_std=float(np.std(delta)) if full else np.nan,
        delta_log_var=float(np.log10(var)) if var > 0 else np.nan,
        QD_mean=early.QD.mean(),
        QD_median=early.QD.median(),
        QD_std=early.QD.std(),
        QD_slope=slope(change.cycle.to_numpy(), change.QD.to_numpy()),
        QD_slope_ols=float(np.polyfit(change.cycle, change.QD, 1)[0]),
        IR_mean=early.IR.mean(),
        IR_slope=slope(change.cycle.to_numpy(), change.IR.to_numpy()),
        IR_zero_fraction=float(early.IR.eq(0).mean()),
        Tavg_mean=early.Tavg.mean(),
        Tmax_mean=early.Tmax.mean(),
        chargetime_mean=early.chargetime.mean(),
        **current_stats(t, current),
    )


def modeling_table(cells, summary):
    """Day 2 추가 후보를 계산한다. 학습 입력은 별도의 피처 목록으로 선택한다."""
    result = cells.copy()
    result["newstructure"] = result.policy.str.contains(
        "newstructure", regex=False
    ).astype(int)
    result["policy_interaction"] = result.rate1_C * result.switch_pct / 100
    clean = summary.copy()
    clean.loc[flag_qd(summary), "QD"] = np.nan
    pivot = clean[clean.cycle.isin([10, 100])].pivot(
        index="cell_id", columns="cycle", values="QD"
    )
    change = pivot[100] - pivot[10]
    result["QD_change"] = result.cell_id.map(change)
    result["QD_relative_change"] = result.cell_id.map(change / pivot[10].where(pivot[10].gt(0)))
    return result


def auxiliary_features(cells, summary):
    """IR 처리 가정과 Day 1에서 계획한 추가 피처를 준비한다."""
    result = modeling_table(cells, summary)
    early = summary[summary.cycle.between(2, 100) & ~summary.placeholder].copy()
    # IR의 0을 결측으로 보는 보조 실험이다. 실제 미측정 여부를 확정하지 않는다.
    early["IR_nonzero"] = early.IR.where(early.IR.gt(0))
    means = early.groupby("cell_id").IR_nonzero.mean()
    slopes = {}
    for cid, group in early[early.cycle.between(10, 100)].groupby("cell_id"):
        slopes[cid] = slope(group.cycle.to_numpy(), group.IR_nonzero.to_numpy())
    result["IR_nonzero_mean"] = result.cell_id.map(means)
    result["IR_nonzero_slope"] = result.cell_id.map(slopes)
    result["IR_all_zero"] = result.IR_zero_fraction.eq(1).astype(int)
    return result
