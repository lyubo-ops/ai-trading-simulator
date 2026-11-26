import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ai_engine.features import FEATURE_COLUMNS, prepare_ml_dataset
from ai_engine.validator import generate_walk_forward_splits

SEED = 42


def generate_ai_predictions(
    df_ohlcv: pd.DataFrame,
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    dataset = prepare_ml_dataset(df_ohlcv)
    splits = generate_walk_forward_splits(
        dataset, train_years=train_years, test_months=test_months
    )

    if not splits:
        raise ValueError("Failed to generate any walk-forward splits.")

    oos_frames: list[pd.DataFrame] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for train_df, test_df in splits:
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["target"]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["target"]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=SEED,
        )
        model.fit(X_train, y_train)

        if len(model.classes_) == 2:
            class_1_idx = list(model.classes_).index(1)
            probs = model.predict_proba(X_test)[:, class_1_idx]
        else:
            only_class = int(model.classes_[0])
            probs = [1.0 if only_class == 1 else 0.0] * len(X_test)

        preds = [1 if p >= 0.5 else 0 for p in probs]

        fold_result = test_df[["date", "open", "high", "low", "close"]].copy()
        fold_result["prob_up"] = probs
        oos_frames.append(fold_result)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(preds)

    signals_df = pd.concat(oos_frames, ignore_index=True)

    ml_metrics = {
        "accuracy": round(float(accuracy_score(all_y_true, all_y_pred)), 4),
        "precision": round(float(precision_score(all_y_true, all_y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(all_y_true, all_y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(all_y_true, all_y_pred, zero_division=0)), 4),
    }

    return {
        "signals_df": signals_df,
        "ml_metrics": ml_metrics,
    }