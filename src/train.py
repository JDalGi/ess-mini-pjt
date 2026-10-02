"""Batch 1 정책 그룹 검증으로 모델을 선택하고 Batch 2에서 최종 평가한다."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_GROUPS
from .preprocess import ROOT


def policy_groups(cells):
    def key(row):
        rates = row[["rate1_C", "switch_pct", "rate2_C"]]
        return (
            "|".join(format(float(v), ".12g") for v in rates)
            if rates.notna().all()
            else row.policy
        )

    return cells.apply(key, axis=1)


def metrics(y, prediction):
    return dict(
        MAPE=100 * mean_absolute_percentage_error(y, prediction),
        MAE=mean_absolute_error(y, prediction),
        RMSE=np.sqrt(mean_squared_error(y, prediction)),
    )


def candidates():
    return {
        "Median": (DummyRegressor(strategy="median"), {}),
        "Linear": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"regressor__model__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "ElasticNet": (
            ElasticNet(max_iter=20000),
            {
                "regressor__model__alpha": [0.001, 0.01, 0.1],
                "regressor__model__l1_ratio": [0.2, 0.8],
            },
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=100, random_state=28),
            {
                "regressor__model__max_depth": [2, 4],
                "regressor__model__min_samples_leaf": [5, 10],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=28),
            {
                "regressor__model__max_depth": [1, 2],
                "regressor__model__n_estimators": [30, 60],
            },
        ),
    }


# Day 1의 정성적 선택 기준을 이번 재실험 전에 수치로 고정한다.
MIN_IMPROVEMENT_PP = 1.0


def pow10(values):
    return np.power(10.0, values)


def fixed_estimator(name, log_target, parameters=None):
    """전처리는 학습 폴드 안에서 적합하며 로그 예측은 사이클 단위로 복원한다."""
    model, _ = candidates()[name]
    estimator = TransformedTargetRegressor(
        regressor=Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", clone(model)),
        ]),
        func=np.log10 if log_target else None,
        inverse_func=pow10 if log_target else None,
    )
    return estimator.set_params(**(parameters or {}))


def select_candidates(selection, min_improvement_pp=MIN_IMPROVEMENT_PP):
    """같은 모델·타깃의 기준 후보와 모든 폴드에서 비교한다."""
    from .features import GROUP_PARENTS
    result = selection.copy()
    result["eligible"] = False
    result["reference"] = ""
    result["improvement_pp"] = np.nan
    result["improved_folds"] = 0
    result["selection_reason"] = ""
    for index, row in result.iterrows():
        parent = GROUP_PARENTS.get(row.bundle)
        if parent is None:
            result.loc[index, ["eligible", "selection_reason"]] = [True, "기본 또는 대표 피처 교체 후보"]
            continue
        matches = result[(result.bundle == parent) & (result.model == row.model)
                         & (result.log_target == row.log_target)]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous parent for {row.bundle}")
        reference = matches.iloc[0]
        current_scores = np.asarray(row.fold_MAPE, dtype=float)
        ref_scores = np.asarray(reference.fold_MAPE, dtype=float)
        improvement = float(np.mean(ref_scores - current_scores))
        improved = int(np.sum(current_scores < ref_scores))
        passes = bool(reference.eligible and improvement >= min_improvement_pp
                      and improved == len(current_scores))
        result.loc[index, ["eligible", "reference", "improvement_pp", "improved_folds", "selection_reason"]] = [
            passes, parent, improvement, improved,
            "평균·전 폴드 개선 기준 충족" if passes else "개선 크기·일관성 또는 부모 후보 채택 기준 미충족",
        ]
    eligible = result[result.eligible]
    if eligible.empty:
        raise ValueError("No eligible candidate")
    # 점수가 같은 경우 피처 수와 모델 복잡도가 작은 후보를 우선한다.
    order = {name: i for i, name in enumerate(candidates())}
    chosen = eligible.assign(model_order=eligible.model.map(order)).sort_values(
        ["CV_MAPE", "n_features", "model_order"], kind="stable"
    ).index[0]
    result["selected"] = result.index == chosen
    return result, chosen


def performance_report(performance):
    notes = {
        "Train (Batch 1 CV)": "모델 선택에 사용한 그룹 CV 평균; 독립 성능 추정 아님",
        "Valid (Batch 1 Hold-out)": "CV와 정책 그룹 중복 없이 분리; 재조정에 사용하지 않음",
        "Test (Batch 2)": "설정 고정 후 Batch 1 전체 재학습",
        "Gap (Train-Valid)": "Valid−Train (%p); (+): 과적합 의심",
        "Gap (Valid-Test)": "Test−Valid (%p); (+): 배치 간 일반화 저하 의심",
        "Gap (Target-Test)": "Test−9.1 (%p); Target: 원논문 비교 목표 9.1%",
        "Test (Batch 3)": "선택적 추가 배치 평가",
        "Gap (Batch2-Batch3)": "Batch 3−Batch 2 (%p)",
        "Gap (Target-Test Batch 3)": "Batch 3−9.1 (%p)",
    }
    return pd.DataFrame({"구분": performance.Index, "MAPE (%)": performance.MAPE,
                         "비고": performance.Index.map(notes)})


def run_modeling(cells, output_dir=ROOT / "results", include_batch3=False, random_state=28):
    """Batch 1 CV만으로 선택을 완료한 뒤 Hold-out과 Batch 2를 평가한다."""
    import json
    import joblib
    from .features import SEARCH_GROUPS, FORBIDDEN_FEATURES
    if not cells.cell_id.is_unique:
        raise ValueError("Input must contain one row per unique battery cell")
    required = set(c for cols in SEARCH_GROUPS.values() for c in cols)
    if required & FORBIDDEN_FEATURES:
        raise ValueError("Future or target information found in feature registry")
    if required - set(cells.columns):
        raise ValueError(f"Missing early features: {sorted(required - set(cells.columns))}")
    labeled = cells[cells.cycle_life.notna()].copy()
    if not np.isfinite(labeled.cycle_life).all() or labeled.cycle_life.le(100).any():
        raise ValueError("Life must be finite and exceed the prediction horizon (100 cycles)")
    batch1 = labeled[labeled.batch.eq("Batch 1")].reset_index(drop=True)
    if batch1.empty or labeled[labeled.batch.eq("Batch 2")].empty:
        raise ValueError("Both Batch 1 and Batch 2 are required")
    groups = policy_groups(batch1)
    fit_idx, valid_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
        random_state=random_state).split(batch1, groups=groups))
    fitting, valid = batch1.iloc[fit_idx], batch1.iloc[valid_idx]
    fit_groups = groups.iloc[fit_idx]
    assert not set(fit_groups) & set(groups.iloc[valid_idx])
    splits = min(5, fit_groups.nunique())
    if splits < 2:
        raise ValueError("At least two training policy groups are required")
    cv = list(GroupKFold(n_splits=splits).split(fitting, groups=fit_groups))
    fold_assignments = []
    for fold, (training, testing) in enumerate(cv, 1):
        assert not set(fit_groups.iloc[training]) & set(fit_groups.iloc[testing])
        for idx, role in [(training, "train"), (testing, "valid")]:
            fold_assignments.append(fitting.iloc[idx][["cell_id"]].assign(
                policy_group=fit_groups.iloc[idx].to_numpy(), fold=fold, role=role))
    rows, estimators = [], []
    for bundle, columns in SEARCH_GROUPS.items():
        print(f"CV feature bundle: {bundle}", flush=True)
        for name, (_, grid) in candidates().items():
            if name == "Median" and bundle != "A":
                continue
            for log_target in [False, True]:
                search = GridSearchCV(fixed_estimator(name, log_target), grid,
                    scoring="neg_mean_absolute_percentage_error", cv=cv,
                    n_jobs=1, error_score="raise")
                search.fit(fitting[columns], fitting.cycle_life)
                scores = [-100 * search.cv_results_[f"split{i}_test_score"][search.best_index_]
                          for i in range(splits)]
                rows.append(dict(bundle=bundle, model=name, log_target=log_target,
                    CV_MAPE=float(np.mean(scores)), CV_std=float(np.std(scores)),
                    fold_MAPE=scores, n_features=len(columns), parameters=str(search.best_params_)))
                estimators.append(search.best_estimator_)
    selection, chosen = select_candidates(pd.DataFrame(rows))
    best = selection.loc[chosen]
    columns = SEARCH_GROUPS[best.bundle]
    selected = estimators[chosen]
    fold_scores = []
    for fold_number, (train_idx, test_idx) in enumerate(cv, 1):
        fold = clone(selected).fit(fitting.iloc[train_idx][columns], fitting.iloc[train_idx].cycle_life)
        fold_scores.append(dict(fold=fold_number, **metrics(fitting.iloc[test_idx].cycle_life,
            fold.predict(fitting.iloc[test_idx][columns]))))
    valid_pred = selected.predict(valid[columns])
    performance = [dict(Index="Train (Batch 1 CV)",
        **pd.DataFrame(fold_scores)[["MAPE", "MAE", "RMSE"]].mean().to_dict()),
        dict(Index="Valid (Batch 1 Hold-out)", **metrics(valid.cycle_life, valid_pred))]
    final = clone(selected).fit(batch1[columns], batch1.cycle_life)
    predictions = [valid[["batch", "cell_id", "cycle_life"]].assign(split="Hold-out", prediction=valid_pred)]
    for label in ["Batch 2"] + (["Batch 3"] if include_batch3 else []):
        testing = labeled[labeled.batch.eq(label)]
        if testing.empty:
            raise ValueError(f"No labeled cells for {label}")
        pred = final.predict(testing[columns])
        performance.append(dict(Index=f"Test ({label})", **metrics(testing.cycle_life, pred)))
        predictions.append(testing[["batch", "cell_id", "cycle_life"]].assign(split="Test", prediction=pred))
    train, holdout, test = performance[:3]
    for label, value in [("Gap (Train-Valid)", holdout["MAPE"]-train["MAPE"]),
                         ("Gap (Valid-Test)", test["MAPE"]-holdout["MAPE"]),
                         ("Gap (Target-Test)", test["MAPE"]-9.1)]:
        performance.append(dict(Index=label, MAPE=value, MAE=np.nan, RMSE=np.nan))
    if include_batch3:
        b3 = performance[3]["MAPE"]
        performance.extend([dict(Index="Gap (Batch2-Batch3)", MAPE=b3-test["MAPE"]),
                            dict(Index="Gap (Target-Test Batch 3)", MAPE=b3-9.1)])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    performance = pd.DataFrame(performance)
    performance.to_csv(output_dir/"model_performance.csv", index=False)
    performance_report(performance).to_csv(output_dir/"performance_report.csv", index=False)
    selection.to_csv(output_dir/"model_selection.csv", index=False)
    pd.concat(predictions).to_csv(output_dir/"predictions.csv", index=False)
    batch1[["cell_id"]].assign(policy_group=groups,
        split=np.where(batch1.index.isin(valid_idx), "Hold-out", "CV")).to_csv(output_dir/"batch1_split.csv", index=False)
    pd.concat(fold_assignments).to_csv(output_dir/"cv_assignments.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(output_dir/"selected_cv_folds.csv", index=False)
    pd.DataFrame([dict(bundle=best.bundle, model=best.model, log_target=best.log_target,
        features=", ".join(columns), parameters=best.parameters, selection_reason=best.selection_reason)]).to_csv(
        output_dir/"selected_model.csv", index=False)
    joblib.dump({"model": final, "features": columns, "prediction_cycle": 100}, output_dir/"model.joblib")
    (output_dir/"run_metadata.json").write_text(json.dumps(dict(random_state=random_state,
        prediction_cycle=100, n_cv_cells=len(fitting), n_holdout_cells=len(valid),
        cv_folds=splits, min_improvement_pp=MIN_IMPROVEMENT_PP,
        required_improved_folds=splits, candidate_count=len(selection),
        selection_uses="Batch 1 policy-group CV only", model_selection_cv_is_independent=False,
        historical_batch2_results_already_seen=True, include_batch3=include_batch3,
        target_mape_pct=9.1), ensure_ascii=False, indent=2))
    return performance, final, selection


def run_auxiliary_experiments(cells, output_dir=ROOT / "results" / "auxiliary",
                              model_results_dir=ROOT / "results"):
    """Day 1 전략과 연결되는 세 가지 보조 비교를 수행한다.

    최종 모델 설정과 정책 그룹 분할을 유지해 IR·충전시간·OLS 처리와
    종료 용량 표식의 민감도를 확인한다. 최종 모델을 다시 선택하지 않는다.
    """
    import ast

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled = cells[cells.cycle_life.notna()]
    batch1 = labeled[labeled.batch.eq("Batch 1")]
    batch2 = labeled[labeled.batch.eq("Batch 2")]
    model_results_dir = Path(model_results_dir)
    split = pd.read_csv(model_results_dir / "batch1_split.csv")
    fitting = batch1[
        batch1.cell_id.isin(split.loc[split.split.eq("CV"), "cell_id"])
    ].reset_index(drop=True)
    valid = batch1[
        batch1.cell_id.isin(split.loc[split.split.eq("Hold-out"), "cell_id"])
    ]
    groups = policy_groups(fitting)
    cv = list(
        GroupKFold(n_splits=min(5, groups.nunique())).split(fitting, groups=groups)
    )
    selected = pd.read_csv(model_results_dir / "selected_model.csv").iloc[0]
    estimator = fixed_estimator(
        selected.model, bool(selected.log_target), ast.literal_eval(selected.parameters)
    )
    base = FEATURE_GROUPS["B"]
    variants = {
        "A": FEATURE_GROUPS["A"],
        "B": base,
        "B_QD_change": base + ["QD_change"],
        "B_QD_relative": base + ["QD_relative_change"],
        "A_IR_raw": FEATURE_GROUPS["A"] + ["IR_mean"],
        "A_IR_nonzero": FEATURE_GROUPS["A"] + ["IR_nonzero_mean", "IR_all_zero"],
        "A_chargetime": FEATURE_GROUPS["A"] + ["chargetime_mean"],
        "B_OLS": ["delta_log_var", "QD_median", "QD_slope_ols"],
        "endpoint_reference": selected.features.split(", "),
        "endpoint_excluded": selected.features.split(", "),
    }
    rows, fold_rows, predictions = [], [], []
    for variant, columns in variants.items():
        endpoint_pair = variant.startswith("endpoint_")
        exclude = variant == "endpoint_excluded"
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv, 1):
            training = fitting.iloc[train_idx]
            testing = fitting.iloc[valid_idx]
            # 10셀 제외 비교는 같은 비표식 검증 셀에서 평가한다.
            if endpoint_pair:
                testing = testing[~testing.endpoint_high_capacity.astype(bool)]
            if exclude:
                training = training[~training.endpoint_high_capacity.astype(bool)]
            if testing.empty:
                continue
            fitted = clone(estimator).fit(training[columns], training.cycle_life)
            score = metrics(testing.cycle_life, fitted.predict(testing[columns]))
            scores.append(score["MAPE"])
            fold_rows.append(
                dict(
                    variant=variant,
                    fold=fold,
                    n_train=len(training),
                    n_valid=len(testing),
                    **score,
                )
            )
        training = (
            fitting[~fitting.endpoint_high_capacity.astype(bool)]
            if exclude
            else fitting
        )
        testing = (
            valid[~valid.endpoint_high_capacity.astype(bool)]
            if endpoint_pair
            else valid
        )
        fitted = clone(estimator).fit(training[columns], training.cycle_life)
        holdout = metrics(testing.cycle_life, fitted.predict(testing[columns]))
        refit = (
            batch1[~batch1.endpoint_high_capacity.astype(bool)] if exclude else batch1
        )
        final = clone(estimator).fit(refit[columns], refit.cycle_life)
        prediction = final.predict(batch2[columns])
        score = metrics(batch2.cycle_life, prediction)
        predictions.append(
            batch2[["cell_id", "cycle_life"]].assign(
                variant=variant, prediction=prediction
            )
        )
        rows.append(
            dict(
                variant=variant,
                CV_MAPE=float(np.mean(scores)),
                CV_std=float(np.std(scores)),
                Holdout_MAPE=holdout["MAPE"],
                n_holdout=len(testing),
                n_refit=len(refit),
                Batch2_MAPE=score["MAPE"],
                Batch2_MAE=score["MAE"],
            )
        )
    comparison = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    paired = []
    pairs = [
        ("B", "A"),
        ("B_QD_change", "B"),
        ("B_QD_relative", "B"),
        ("A_IR_raw", "A"),
        ("A_IR_nonzero", "A_IR_raw"),
        ("A_chargetime", "A"),
        ("B_OLS", "B"),
        ("endpoint_excluded", "endpoint_reference"),
    ]
    for variant, reference in pairs:
        current = folds[folds.variant.eq(variant)]
        baseline = folds[folds.variant.eq(reference)]
        joined = current.merge(baseline, on="fold", suffixes=("_current", "_reference"))
        gap = joined.MAPE_current - joined.MAPE_reference
        paired.append(
            dict(
                variant=variant,
                reference=reference,
                improved_folds=int(gap.lt(0).sum()),
                total_folds=len(gap),
                mean_gap_pp=gap.mean(),
            )
        )
    paired = pd.DataFrame(paired)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    folds.to_csv(output_dir / "folds.csv", index=False)
    paired.to_csv(output_dir / "paired_folds.csv", index=False)
    pd.concat(predictions).to_csv(output_dir / "predictions.csv", index=False)
    return comparison, folds, paired
