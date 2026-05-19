from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FRAME_HEADER_PREFIX = ("frame", "time_s")
METADATA_COLUMNS = {
    "source_csv",
    "source_video",
    "label_csv",
    "patient_id",
    "session_id",
    "record_time",
    "session_time",
    "time_delta_s",
    "sex",
    "diagnosis",
    "dominant_hand",
    "trial_index",
    "trial_role",
    "total_p1_s",
    "total_p2_s",
    "total_s1_s",
    "total_s2_s",
}


@dataclass
class LabelRecord:
    patient_id: str
    session_id: str
    session_time: Optional[datetime]
    sex: str
    diagnosis: str
    dominant_hand: str
    label_csv: str
    totals: Dict[str, float]


@dataclass
class TrackingRecord:
    csv_path: Path
    source_video: str
    patient_id: Optional[str]
    record_time: Optional[datetime]
    camera_id: Optional[int]


def normalize_patient_id(text: str) -> Optional[str]:
    match = re.search(r"patient[_-]?(\d+)", text, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"pati\w*[_-]?(\d+)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return f"patient_{int(match.group(1)):03d}"


def parse_datetime_from_text(text: str) -> Optional[datetime]:
    match = re.search(r"(20\d{6})_(\d{2})_(\d{2})_(\d{2})", text)
    if match:
        raw = "".join(match.groups())
        return datetime.strptime(raw, "%Y%m%d%H%M%S")

    for raw in re.findall(r"20\d{10,12}", text):
        if len(raw) == 14:
            return datetime.strptime(raw, "%Y%m%d%H%M%S")
        if len(raw) == 12:
            return datetime.strptime(raw, "%Y%m%d%H%M")
    return None


def parse_camera_id(text: str) -> Optional[int]:
    match = re.search(r"camP_(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def safe_float(value: object) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_label_records(data_root: Path) -> List[LabelRecord]:
    labels: List[LabelRecord] = []
    for path in sorted(data_root.rglob("*MS*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))

        meta = next((row for row in rows if len(row) >= 8 and row[4] in {"Male", "Female"}), None)
        if meta is None:
            continue

        patient_id = normalize_patient_id(str(path.parent)) or normalize_patient_id(path.name)
        if patient_id is None:
            continue

        session_time = parse_datetime_from_text(meta[7]) or parse_datetime_from_text(path.name)
        session_id = f"{patient_id}_{session_time:%Y%m%d%H%M%S}" if session_time else f"{patient_id}_{path.stem}"

        totals_row = next((row for row in rows if row and row[0] == "9"), [])
        totals = {
            "total_p1_s": safe_float(totals_row[2] if len(totals_row) > 2 else None),
            "total_p2_s": safe_float(totals_row[4] if len(totals_row) > 4 else None),
            "total_s1_s": safe_float(totals_row[6] if len(totals_row) > 6 else None),
            "total_s2_s": safe_float(totals_row[8] if len(totals_row) > 8 else None),
        }

        labels.append(
            LabelRecord(
                patient_id=patient_id,
                session_id=session_id,
                session_time=session_time,
                sex=meta[4],
                diagnosis=meta[5],
                dominant_hand=meta[6].lower(),
                label_csv=str(path),
                totals=totals,
            )
        )
    return labels


def is_frame_level_tracking_csv(path: Path) -> bool:
    if "MS" in path.name or path.name.startswith("batch_summary"):
        return False
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
    except (OSError, StopIteration, UnicodeDecodeError):
        return False
    return tuple(header[:2]) == FRAME_HEADER_PREFIX


def read_source_video_from_sibling_json(csv_path: Path) -> Optional[str]:
    candidates = sorted(csv_path.parent.glob("*summary*.json")) + sorted(csv_path.parent.glob("*.json"))
    for json_path in candidates:
        try:
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        video = data.get("video") if isinstance(data, dict) else None
        if video:
            return str(video)
    return None


def collect_tracking_records(roots: Sequence[Path], camera_filter: Optional[int]) -> List[TrackingRecord]:
    records: List[TrackingRecord] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            resolved = csv_path.resolve()
            if resolved in seen or ".venv" in resolved.parts:
                continue
            seen.add(resolved)

            if not is_frame_level_tracking_csv(csv_path):
                continue

            source_video = read_source_video_from_sibling_json(csv_path)
            if source_video is None:
                same_stem_video = csv_path.with_suffix(".mp4")
                source_video = str(same_stem_video if same_stem_video.exists() else csv_path)

            camera_id = parse_camera_id(source_video) or parse_camera_id(str(csv_path))
            if camera_filter is not None and camera_id != camera_filter:
                continue

            text_for_ids = f"{source_video} {csv_path}"
            records.append(
                TrackingRecord(
                    csv_path=csv_path,
                    source_video=source_video,
                    patient_id=normalize_patient_id(text_for_ids),
                    record_time=parse_datetime_from_text(text_for_ids),
                    camera_id=camera_id,
                )
            )

    return records


def match_label(
    record: TrackingRecord,
    labels_by_patient: Dict[str, List[LabelRecord]],
    match_window_minutes: float,
) -> Tuple[Optional[LabelRecord], float]:
    if record.patient_id is None or record.patient_id not in labels_by_patient:
        return None, math.nan

    candidates = labels_by_patient[record.patient_id]
    if record.record_time is None:
        return (candidates[0], math.nan) if len(candidates) == 1 else (None, math.nan)

    scored: List[Tuple[float, LabelRecord]] = []
    for label in candidates:
        if label.session_time is None:
            continue
        delta_s = (record.record_time - label.session_time).total_seconds()
        if abs(delta_s) <= match_window_minutes * 60.0:
            scored.append((abs(delta_s), label))

    if not scored:
        return None, math.nan

    _, best_label = min(scored, key=lambda item: item[0])
    delta_s = (record.record_time - best_label.session_time).total_seconds() if best_label.session_time else math.nan
    return best_label, float(delta_s)


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def add_stats(features: Dict[str, float], prefix: str, values: Iterable[float]) -> None:
    series = pd.Series(list(values), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        for suffix in ["mean", "std", "median", "p10", "p25", "p75", "p90", "max", "min", "iqr"]:
            features[f"{prefix}_{suffix}"] = math.nan
        return

    features[f"{prefix}_mean"] = float(series.mean())
    features[f"{prefix}_std"] = float(series.std(ddof=0))
    features[f"{prefix}_median"] = float(series.median())
    features[f"{prefix}_p10"] = float(series.quantile(0.10))
    features[f"{prefix}_p25"] = float(series.quantile(0.25))
    features[f"{prefix}_p75"] = float(series.quantile(0.75))
    features[f"{prefix}_p90"] = float(series.quantile(0.90))
    features[f"{prefix}_max"] = float(series.max())
    features[f"{prefix}_min"] = float(series.min())
    features[f"{prefix}_iqr"] = float(series.quantile(0.75) - series.quantile(0.25))


def choose_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        ("ma_x_mm", "ma_y_mm"),
        ("hand_x_mm", "hand_y_mm"),
        ("ema_x_mm", "ema_y_mm"),
        ("kalman_x_px", "kalman_y_px"),
        ("ema_x_px", "ema_y_px"),
        ("WRIST_x", "WRIST_y"),
    ]
    for x_col, y_col in candidates:
        if x_col in df and y_col in df:
            return x_col, y_col
    return None, None


def extract_features_from_csv(record: TrackingRecord, label: LabelRecord, time_delta_s: float) -> Dict[str, object]:
    df = pd.read_csv(record.csv_path)
    features: Dict[str, object] = {
        "source_csv": str(record.csv_path),
        "source_video": record.source_video,
        "label_csv": label.label_csv,
        "patient_id": label.patient_id,
        "session_id": label.session_id,
        "record_time": record.record_time.isoformat() if record.record_time else "",
        "session_time": label.session_time.isoformat() if label.session_time else "",
        "time_delta_s": time_delta_s,
        "sex": label.sex,
        "diagnosis": label.diagnosis,
        "dominant_hand": label.dominant_hand,
        "camera_id": record.camera_id if record.camera_id is not None else -1,
        **label.totals,
    }

    features["frame_count"] = int(len(df))
    time_s = finite_series(df, "time_s")
    features["duration_s"] = float(time_s.max() - time_s.min()) if not time_s.empty else math.nan
    features["fps_est"] = float((len(time_s) - 1) / features["duration_s"]) if features["duration_s"] and features["duration_s"] > 0 else math.nan

    if "hand_detected" in df:
        detected = finite_series(df, "hand_detected")
        features["hand_detection_rate"] = float(detected.mean()) if not detected.empty else math.nan
    else:
        x_col, _ = choose_xy_columns(df)
        features["hand_detection_rate"] = float(finite_series(df, x_col).notna().mean()) if x_col else math.nan

    x_col, y_col = choose_xy_columns(df)
    if x_col and y_col:
        x = finite_series(df, x_col)
        y = finite_series(df, y_col)
        n = min(len(x), len(y))
        x = x.iloc[:n].reset_index(drop=True)
        y = y.iloc[:n].reset_index(drop=True)

        features["x_range"] = float(x.max() - x.min()) if n else math.nan
        features["y_range"] = float(y.max() - y.min()) if n else math.nan
        features["x_std"] = float(x.std(ddof=0)) if n else math.nan
        features["y_std"] = float(y.std(ddof=0)) if n else math.nan

        if n >= 2:
            dx = x.diff().iloc[1:]
            dy = y.diff().iloc[1:]
            step = np.sqrt(dx.to_numpy() ** 2 + dy.to_numpy() ** 2)
            path = float(np.nansum(step))
            displacement = float(math.hypot(x.iloc[-1] - x.iloc[0], y.iloc[-1] - y.iloc[0]))
            features["path_from_xy"] = path
            features["displacement"] = displacement
            features["straightness"] = displacement / path if path > 0 else math.nan
        else:
            features["path_from_xy"] = math.nan
            features["displacement"] = math.nan
            features["straightness"] = math.nan

    if "cumulative_distance_mm" in df:
        cumulative = finite_series(df, "cumulative_distance_mm")
        features["total_distance_mm"] = float(cumulative.max()) if not cumulative.empty else math.nan

    speed_columns = [col for col in ["speed_ema_mm_s", "speed_mm_s"] if col in df]
    if not speed_columns:
        speed_columns = [col for col in df.columns if col.endswith("_speed_px_s")]
    speed_values = pd.concat([finite_series(df, col) for col in speed_columns], ignore_index=True) if speed_columns else pd.Series(dtype=float)
    add_stats(features, "speed", speed_values)

    accel = finite_series(df, "acceleration_mm_s2").abs()
    if accel.empty and not speed_values.empty and not time_s.empty:
        dt = float(time_s.diff().median()) if len(time_s) > 1 else math.nan
        if dt and dt > 0:
            accel = speed_values.diff().abs() / dt
    add_stats(features, "abs_accel", accel)

    if {"THUMB_TIP_x", "THUMB_TIP_y", "INDEX_FINGER_TIP_x", "INDEX_FINGER_TIP_y"}.issubset(df.columns):
        thumb_x = pd.to_numeric(df["THUMB_TIP_x"], errors="coerce")
        thumb_y = pd.to_numeric(df["THUMB_TIP_y"], errors="coerce")
        index_x = pd.to_numeric(df["INDEX_FINGER_TIP_x"], errors="coerce")
        index_y = pd.to_numeric(df["INDEX_FINGER_TIP_y"], errors="coerce")
        pinch = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
        add_stats(features, "pinch_distance", pinch)

    for col in ["pin_count", "occupied_count"]:
        if col in df:
            add_stats(features, col, finite_series(df, col))

    return features


def add_trial_roles(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["trial_index"] = np.nan
    features["trial_role"] = ""

    if "record_time" not in features or "session_id" not in features:
        return features

    for session_id, group in features.groupby("session_id", dropna=False):
        times = sorted(t for t in group["record_time"].dropna().unique() if str(t))
        if not times:
            continue
        index_by_time = {time: idx for idx, time in enumerate(times)}
        for row_idx, row in group.iterrows():
            trial_index = index_by_time.get(row["record_time"])
            if trial_index is None:
                continue
            features.at[row_idx, "trial_index"] = trial_index
            features.at[row_idx, "trial_role"] = "dominant" if trial_index < 2 else "nondominant"

    return features


def build_feature_table(args: argparse.Namespace) -> pd.DataFrame:
    labels = read_label_records(Path(args.data_root))
    labels_by_patient: Dict[str, List[LabelRecord]] = {}
    for label in labels:
        labels_by_patient.setdefault(label.patient_id, []).append(label)

    for patient_labels in labels_by_patient.values():
        patient_labels.sort(key=lambda item: item.session_time or datetime.min)

    tracking_roots = [Path(root) for root in args.tracking_root]
    camera_filter = None if args.camera == "all" else int(args.camera)
    records = collect_tracking_records(tracking_roots, camera_filter=camera_filter)

    selected_patients: Optional[set[str]] = None
    if args.max_patients > 0:
        patients = sorted({record.patient_id for record in records if record.patient_id})
        selected_patients = set(patients[: args.max_patients])

    rows: List[Dict[str, object]] = []
    skipped = 0
    for record in records:
        if selected_patients is not None and record.patient_id not in selected_patients:
            continue
        label, time_delta_s = match_label(record, labels_by_patient, args.match_window_minutes)
        if label is None:
            skipped += 1
            continue
        try:
            rows.append(extract_features_from_csv(record, label, time_delta_s))
        except Exception as exc:
            skipped += 1
            print(f"Skipping {record.csv_path}: {exc}")

    if not rows:
        raise RuntimeError("No labelled tracking CSV files were found. Generate tracking outputs first.")

    features = add_trial_roles(pd.DataFrame(rows))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"Feature table: {output}")
    print(f"Rows={len(features)} patients={features['patient_id'].nunique()} skipped={skipped}")
    return features


def numeric_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    blocked = set(METADATA_COLUMNS)
    blocked.add(target)
    numeric_cols = []
    for col in df.columns:
        if col in blocked:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def make_train_test_masks(
    df: pd.DataFrame,
    target: str,
    train_patients: int,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    groups = df["patient_id"].astype(str).to_numpy()

    if train_patients > 0:
        rng = np.random.default_rng(random_state)
        patients = np.array(sorted(df["patient_id"].astype(str).unique()))
        rng.shuffle(patients)
        train_set = set(patients[: min(train_patients, len(patients))])
        train_mask = df["patient_id"].astype(str).isin(train_set).to_numpy()
        test_mask = ~train_mask
        if test_mask.any():
            return train_mask, test_mask

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df[target], groups=groups))
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def train_model(args: argparse.Namespace) -> Dict[str, object]:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
        )
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Training requires a working scikit-learn/scipy installation. "
            "Install dependencies from requirements-local.txt in a clean venv."
        ) from exc

    df = pd.read_csv(args.features)
    target = "dominant_hand" if args.task == "dominant_side" else "trial_role"
    df = df[df[target].notna() & (df[target].astype(str) != "")].copy()

    if df.empty:
        raise RuntimeError(f"No rows with target '{target}' found in {args.features}")

    feature_cols = numeric_feature_columns(df, target)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found.")

    y = df[target].astype(str)
    if y.nunique() < 2:
        raise RuntimeError(
            f"Need at least two target classes for training, found {sorted(y.unique())}. "
            "Generate tracking CSVs for more patients/classes first."
        )

    train_mask, test_mask = make_train_test_masks(
        df,
        target=target,
        train_patients=args.train_patients,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_classes = sorted(y[train_mask].unique())
    test_classes = sorted(y[test_mask].unique())
    if len(train_classes) < 2 or len(test_classes) < 2:
        raise RuntimeError(
            f"Train/test split is not usable. Train classes={train_classes}, test classes={test_classes}. "
            "Use more patients or change --train-patients/--test-size."
        )

    x_train = df.loc[train_mask, feature_cols].apply(pd.to_numeric, errors="coerce")
    x_test = df.loc[test_mask, feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = y[train_mask]
    y_test = y[test_mask]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.trees,
                    min_samples_leaf=args.min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "task": args.task,
        "target": target,
        "rows": int(len(df)),
        "features": feature_cols,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_patients": int(df.loc[train_mask, "patient_id"].nunique()),
        "test_patients": int(df.loc[test_mask, "patient_id"].nunique()),
        "classes": sorted(y.unique()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=sorted(y.unique())).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    classifier = model.named_steps["classifier"]
    importances = sorted(
        zip(feature_cols, classifier.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    metrics["top_features"] = [{"feature": name, "importance": float(score)} for name, score in importances[:20]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.task}_model.joblib"
    metrics_path = output_dir / f"{args.task}_metrics.json"
    predictions_path = output_dir / f"{args.task}_test_predictions.csv"

    joblib.dump({"model": model, "feature_columns": feature_cols, "task": args.task, "target": target}, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df = df.loc[test_mask, ["patient_id", "session_id", "source_csv", target]].copy()
    pred_df["prediction"] = predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        for class_name, values in zip(model.classes_, probabilities.T):
            pred_df[f"prob_{class_name}"] = values
    pred_df.to_csv(predictions_path, index=False)

    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Accuracy={metrics['accuracy']:.3f} balanced_accuracy={metrics['balanced_accuracy']:.3f}")
    return metrics


def predict(args: argparse.Namespace) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Prediction requires joblib. Install requirements-local.txt first.") from exc

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    df = pd.read_csv(args.features)
    x = df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    predictions = model.predict(x)

    out = df[[col for col in ["patient_id", "session_id", "source_csv", "source_video"] if col in df]].copy()
    out["prediction"] = predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        for class_name, values in zip(model.classes_, probabilities.T):
            out[f"prob_{class_name}"] = values

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Predictions: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test AI models for dominant hand analysis from tracking CSVs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-features", help="Create one ML feature row per tracking CSV.")
    build.add_argument("--data-root", default="../data", help="Root containing patient folders and MS*.csv labels.")
    build.add_argument(
        "--tracking-root",
        action="append",
        required=True,
        help="Root containing frame-level tracking CSVs. Can be passed multiple times.",
    )
    build.add_argument("--output", default="outputs_dominant_hand/features.csv")
    build.add_argument("--camera", default="1", help="Camera id to use, or 'all'. Default uses camP_1.")
    build.add_argument("--match-window-minutes", type=float, default=30.0)
    build.add_argument("--max-patients", type=int, default=0, help="Optional deterministic patient limit for experiments.")

    train = subparsers.add_parser("train", help="Train and evaluate a model from a feature table.")
    train.add_argument("--features", default="outputs_dominant_hand/features.csv")
    train.add_argument("--output-dir", default="outputs_dominant_hand")
    train.add_argument("--task", choices=["dominant_side", "dominant_trial"], default="dominant_side")
    train.add_argument("--train-patients", type=int, default=0, help="Use exactly this many patients for training if possible.")
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--trees", type=int, default=400)
    train.add_argument("--min-samples-leaf", type=int, default=2)
    train.add_argument("--n-jobs", type=int, default=-1)
    train.add_argument("--random-state", type=int, default=42)

    all_cmd = subparsers.add_parser("all", help="Build features and train in one command.")
    all_cmd.add_argument("--data-root", default="../data")
    all_cmd.add_argument("--tracking-root", action="append", required=True)
    all_cmd.add_argument("--features", default="outputs_dominant_hand/features.csv")
    all_cmd.add_argument("--camera", default="1")
    all_cmd.add_argument("--match-window-minutes", type=float, default=30.0)
    all_cmd.add_argument("--max-patients", type=int, default=0)
    all_cmd.add_argument("--output-dir", default="outputs_dominant_hand")
    all_cmd.add_argument("--task", choices=["dominant_side", "dominant_trial"], default="dominant_side")
    all_cmd.add_argument("--train-patients", type=int, default=0)
    all_cmd.add_argument("--test-size", type=float, default=0.25)
    all_cmd.add_argument("--trees", type=int, default=400)
    all_cmd.add_argument("--min-samples-leaf", type=int, default=2)
    all_cmd.add_argument("--n-jobs", type=int, default=-1)
    all_cmd.add_argument("--random-state", type=int, default=42)

    pred = subparsers.add_parser("predict", help="Run an existing model on a feature table.")
    pred.add_argument("--model", required=True)
    pred.add_argument("--features", required=True)
    pred.add_argument("--output", default="outputs_dominant_hand/predictions.csv")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-features":
        build_feature_table(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "all":
        build_args = argparse.Namespace(
            data_root=args.data_root,
            tracking_root=args.tracking_root,
            output=args.features,
            camera=args.camera,
            match_window_minutes=args.match_window_minutes,
            max_patients=args.max_patients,
        )
        build_feature_table(build_args)
        train_args = argparse.Namespace(**vars(args))
        train_args.features = args.features
        train_model(train_args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
