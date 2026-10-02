"""세 배치의 MATLAB/HDF5 데이터를 셀별 정보와 사이클 요약표로 읽는다."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
FIELDS = {
    "QDischarge": "QD",
    "QCharge": "QC",
    "IR": "IR",
    "Tmax": "Tmax",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "chargetime": "chargetime",
}
# 전체 139셀의 실제 Vdlin 범위가 2.0~3.5V인 것을 확인했다.
# 같은 전압 구간의 ΔQ 통계를 비교하기 위한 축이며 용량 제거 기준이 아니다.
VOLTAGE_GRID = np.linspace(2.0, 3.5, 1000)


def deref(file: h5py.File, field: h5py.Dataset, i: int) -> np.ndarray:
    dataset = file[field[i, 0]]
    # MATLAB 빈 배열은 차원 정보(예: [0, 0])를 데이터셋에 저장한다.
    # 이 값은 측정값이 아니므로 빈 배열로 반환한다.
    if dataset.attrs.get("MATLAB_empty", 0):
        return np.array([], dtype=float)
    return np.asarray(dataset).ravel()


def decode_chars(values: np.ndarray) -> str:
    return "".join(chr(int(v)) for v in values if int(v) != 0)


def curve_on_grid(
    file: h5py.File, cycles: h5py.Group, voltage: np.ndarray, cycle: int
) -> np.ndarray:
    """Qdlin 곡선을 공통 전압축에 보간한다. 관측 범위 밖으로 외삽하지 않는다."""
    if len(cycles["Qdlin"]) < cycle:
        return np.full(VOLTAGE_GRID.size, np.nan)
    q = deref(file, cycles["Qdlin"], cycle - 1)
    if q.size != voltage.size or q.size < 2:
        return np.full(VOLTAGE_GRID.size, np.nan)
    if not (np.isfinite(q).all() and np.isfinite(voltage).all()):
        return np.full(VOLTAGE_GRID.size, np.nan)
    x, y = voltage, q
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        return np.full(VOLTAGE_GRID.size, np.nan)
    return np.interp(VOLTAGE_GRID, x, y, left=np.nan, right=np.nan)


def read_batch(
    label: str, path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, dict]]:
    """셀 정보, 사이클 요약, ΔQ 곡선과 충전 전류 시계열을 반환한다."""
    # 반환 단위: cells=셀당 한 행, summaries=셀·사이클당 한 행,
    # curves=셀별 ΔQ 배열, current_traces=초기 사이클 내부의 시간·전류.
    # 필요한 곡선만 읽어 대용량 원시 시계열을 전부 메모리에 올리지 않는다.
    cells, summaries, curves, current_traces = [], [], {}, {}
    with h5py.File(path) as file:
        batch = file["batch"]
        for i in range(len(batch["cycle_life"])):
            cell_id = f"{label.replace(' ', '')}_{i:03d}"
            life = float(deref(file, batch["cycle_life"], i)[0])
            policy = decode_chars(deref(file, batch["policy_readable"], i))
            summary = file[batch["summary"][i, 0]]
            cycle_values = np.asarray(summary["cycle"]).ravel()
            if (
                not np.isfinite(cycle_values).all()
                or not np.equal(cycle_values, np.floor(cycle_values)).all()
            ):
                raise ValueError(
                    f"{cell_id}: cycle identifiers must be finite integers"
                )
            cycles = cycle_values.astype(int)
            if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
                raise ValueError(
                    f"{cell_id}: cycle numbering does not match positional traces"
                )
            frame = pd.DataFrame({"cycle": cycles})
            for source, dest in FIELDS.items():
                values = np.asarray(summary[source]).ravel()
                if len(values) != len(cycles):
                    raise ValueError(
                        f"{cell_id}: {source} length differs from cycle length"
                    )
                frame[dest] = values
            frame.insert(0, "cell_id", cell_id)
            frame.insert(0, "batch", label)
            summaries.append(frame)

            cycle_group = file[batch["cycles"][i, 0]]
            frame["is_placeholder"] = False
            zero_rows = frame[list(FIELDS.values())].eq(0).all(axis=1)
            for row_index in frame.index[zero_rows]:
                position = int(frame.loc[row_index, "cycle"]) - 1
                empty_trace = all(
                    deref(file, cycle_group[key], position).size == 0
                    for key in ("I", "t", "V", "Qd")
                )
                frame.loc[row_index, "is_placeholder"] = empty_trace
            voltage = deref(file, batch["Vdlin"], i)
            q10 = curve_on_grid(file, cycle_group, voltage, 10)
            q100 = curve_on_grid(file, cycle_group, voltage, 100)
            curves[cell_id] = q100 - q10
            current_traces[cell_id] = {
                cycle: {
                    key: deref(file, cycle_group[key], cycle - 1)
                    if len(cycle_group[key]) >= cycle
                    else np.array([], dtype=float)
                    for key in ("I", "t")
                }
                for cycle in (10, 100)
            }
            cells.append(
                {
                    "batch": label,
                    "cell_id": cell_id,
                    "cycle_life": life,
                    "policy": policy,
                    "n_cycles": len(cycles),
                    "cycle_min": cycles.min(),
                    "cycle_max": cycles.max(),
                    "voltage_min": np.nanmin(voltage),
                    "voltage_max": np.nanmax(voltage),
                }
            )
    return (
        pd.DataFrame(cells),
        pd.concat(summaries, ignore_index=True),
        curves,
        current_traces,
    )


import re

SIGNALS = FIELDS
GRID = VOLTAGE_GRID


def values(f, field, i):
    ds = f[field[i, 0]]
    return (
        np.array([], dtype=float)
        if ds.attrs.get("MATLAB_empty", 0)
        else np.asarray(ds).ravel()
    )


def load(data_dir=None):
    """원본 배치를 읽어 Day 1의 셀 단위 분석 데이터를 구성한다."""
    from .features import slope, current_stats, knee, early_features

    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
    rows, summaries, curves, traces = [], [], {}, {}
    for batch, filename in FILES.items():
        print(f"Reading {batch}: {filename}", flush=True)
        with h5py.File(data_dir / filename, "r") as f:
            b = f["batch"]
            for j in range(len(b["cycle_life"])):
                cid = f"B{batch[-1]}_{j:03d}"
                s, c = f[b["summary"][j, 0]], f[b["cycles"][j, 0]]
                cy = np.asarray(s["cycle"]).ravel()
                assert np.array_equal(cy, np.arange(1, len(cy) + 1)), (
                    f"{cid}: cycle mapping"
                )
                raw = pd.DataFrame({"cycle": cy.astype(int)})
                for k, v in SIGNALS.items():
                    raw[v] = np.asarray(s[k]).ravel()
                raw.insert(0, "cell_id", cid)
                raw.insert(0, "batch", batch)
                # 모든 요약값이 0인 행은 대응 시계열도 비어 있는지 확인한다.
                placeholder = np.zeros(len(raw), dtype=bool)
                for pos in np.flatnonzero(
                    raw[list(SIGNALS.values())].eq(0).all(axis=1)
                ):
                    placeholder[pos] = all(
                        values(f, c[k], pos).size == 0 for k in ["I", "t", "V", "Qd"]
                    )
                raw["placeholder"] = placeholder
                clean = raw.copy()
                clean[list(SIGNALS.values())] = clean[list(SIGNALS.values())].replace(
                    [np.inf, -np.inf], np.nan
                )
                clean.loc[placeholder, list(SIGNALS.values())] = np.nan
                clean.loc[clean.QD.lt(0), "QD"] = np.nan
                summaries.append(raw)
                early = clean[clean.cycle.between(2, 100)]
                change = early[early.cycle.between(10, 100)]
                policy = "".join(
                    chr(int(v)) for v in values(f, b["policy_readable"], j) if v
                )
                match = re.match(r"^([\d.]+)C\(([\d.]+)%\)-([\d.]+)C(?:$|[-(])", policy)
                rates = tuple(map(float, match.groups())) if match else (np.nan,) * 3
                life = values(f, b["cycle_life"], j)[0]
                voltage = values(f, b["Vdlin"], j)
                assert voltage.size == 1000 and np.isfinite(voltage).all()
                order = np.argsort(voltage)
                assert np.all(np.diff(voltage[order]) > 0)
                qs = []
                for n in [10, 100]:
                    q = values(f, c["Qdlin"], n - 1)
                    if len(q) == len(voltage) and np.isfinite(q).all():
                        qs.append(
                            np.interp(
                                GRID,
                                voltage[order],
                                q[order],
                                left=np.nan,
                                right=np.nan,
                            )
                        )
                    else:
                        qs.append(np.full(len(GRID), np.nan))
                delta = qs[1] - qs[0]
                curves[cid] = delta
                full = np.isfinite(delta).all()
                var = np.var(delta) if full else np.nan
                t, current = values(f, c["t"], 9), values(f, c["I"], 9)
                traces[cid] = (t, current)
                qfinal = float(clean.QD.dropna().iloc[-1])
                initial = float(early[early.cycle.le(10)].QD.median())
                life_curve = clean[clean.cycle.le(life)] if np.isfinite(life) else clean
                late_x = life_curve.cycle.to_numpy()
                late_y = life_curve.QD.to_numpy()
                # 라벨이 있는 셀은 기록 수명까지의 곡선만 사용한다.
                # Batch 2의 라벨 이후 측정을 열화 비교에 섞지 않는다.
                end = float(late_x.max())
                mid = (late_x >= 0.2 * end) & (late_x <= 0.4 * end)
                late = (late_x >= 0.6 * end) & (late_x <= 0.8 * end)
                pre, post = (
                    slope(late_x[mid], late_y[mid]),
                    slope(late_x[late], late_y[late]),
                )
                crossings = clean.loc[clean.QD.le(0.88) & clean.QD.gt(0), "cycle"]
                at_label = clean.loc[clean.cycle.eq(life), "QD"]
                row = dict(
                    batch=batch,
                    cell_id=cid,
                    raw_index=j,
                    policy=policy,
                    cycle_life=float(life),
                    observed_cycles=len(raw),
                    QD_final=qfinal,
                    QD_initial=initial,
                    analysis_last_cycle=end,
                    first_QD_le_088=float(crossings.iloc[0])
                    if len(crossings)
                    else np.nan,
                    QD_at_label=float(at_label.iloc[0]) if len(at_label) else np.nan,
                    label_minus_record_end=float(life - len(raw)),
                    end_relative_capacity=qfinal / initial,
                    # 종료 용량 점검용 표식이다. 관측 종료 QD의 두 집단 사이에
                    # 있는 0.90Ah를 경계로 사용한다. 이 표식만으로 실험 종료
                    # 이유를 확정하거나 EOL을 재정의하고 셀을 삭제하지 않는다.
                    endpoint_high_capacity=bool(qfinal > 0.90),
                    placeholder_rows=int(placeholder.sum()),
                    early_rows=len(early),
                    delta_valid_fraction=float(np.isfinite(delta).mean()),
                    voltage_min=float(voltage.min()),
                    voltage_max=float(voltage.max()),
                    rate1_C=rates[0],
                    switch_pct=rates[1],
                    rate2_C=rates[2],
                    **early_features(early, change, delta, t, current),
                    slope_mid=pre,
                    slope_late=post,
                    accelerates=bool(pre < 0 and post < 2 * pre),
                    **knee(late_x, late_y),
                )
                rows.append(row)
        print(f"{batch} loaded", flush=True)
    return pd.DataFrame(rows), pd.concat(summaries, ignore_index=True), curves, traces
