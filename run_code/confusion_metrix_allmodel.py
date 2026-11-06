import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# =========================================================
# USER SETTINGS
# =========================================================

PROPOSED_CSV = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_fttf_dual_precbiased_gate_selectedfeatures\phase2_fttf_dual_precbiased_gate_selectedfeatures_oof.csv"
LIGHTGBM_CSV = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures\LightGBM_oof.csv"

ALLMODELS_DIR = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures"
OUT_DIR = r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\confusion_matrix_comparison"

THRESHOLD = 0.50


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_file_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{label}] File not found:\n{path}\n\n"
            f"Please check the filename and folder path carefully."
        )


def find_file_by_keywords(folder: str, keywords: list[str], label: str) -> str:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"[{label}] Folder not found:\n{folder}")

    files = sorted(os.listdir(folder))
    matches = []

    for name in files:
        lower = name.lower()
        if lower.endswith(".csv") and all(k.lower() in lower for k in keywords):
            matches.append(os.path.join(folder, name))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"[{label}] No matching CSV found in:\n{folder}\n"
            f"Keywords used: {keywords}\n"
            f"Available files:\n" + "\n".join(files)
        )

    raise ValueError(
        f"[{label}] Multiple matching CSV files found:\n" +
        "\n".join(matches) +
        f"\nPlease choose one explicitly."
    )


def detect_columns(df: pd.DataFrame, file_label: str):
    cols_lower = {c.lower(): c for c in df.columns}

    possible_true = [
        "y_true", "true", "label", "target", "y", "actual",
        "ground_truth", "ytest", "y_test"
    ]
    possible_prob = [
        "y_prob", "prob", "probability", "pred_prob", "p",
        "oof_prob", "positive_prob", "prob_1", "prediction",
        "pred", "score"
    ]

    true_col = None
    prob_col = None

    for name in possible_true:
        if name in cols_lower:
            true_col = cols_lower[name]
            break

    for name in possible_prob:
        if name in cols_lower:
            prob_col = cols_lower[name]
            break

    if true_col is None or prob_col is None:
        raise ValueError(
            f"[{file_label}] Could not automatically detect columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    return true_col, prob_col


def load_oof(csv_path: str, file_label: str) -> pd.DataFrame:
    check_file_exists(csv_path, file_label)

    df = pd.read_csv(csv_path)
    true_col, prob_col = detect_columns(df, file_label)

    out = df[[true_col, prob_col]].copy()
    out.columns = ["y_true", "y_prob"]

    if out["y_true"].isna().any() or out["y_prob"].isna().any():
        raise ValueError(f"[{file_label}] Found missing values in y_true or y_prob.")

    out["y_true"] = out["y_true"].astype(int)
    out["y_prob"] = out["y_prob"].astype(float)

    return out


def validate_alignment(reference_df: pd.DataFrame, compare_df: pd.DataFrame, ref_name: str, cmp_name: str) -> None:
    if len(reference_df) != len(compare_df):
        raise ValueError(
            f"Row count mismatch between {ref_name} ({len(reference_df)}) "
            f"and {cmp_name} ({len(compare_df)})."
        )

    if not np.array_equal(reference_df["y_true"].values, compare_df["y_true"].values):
        raise ValueError(
            f"y_true mismatch between {ref_name} and {cmp_name}.\n"
            f"Make sure all OOF files are aligned row-wise and come from the same CV split."
        )


def get_confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp


def plot_confusion_matrix_panel(ax, cm, title: str):
    import numpy as np

    # Your selected color palette
    colors = {
        (0, 0): "#5b7f2b",  # TN - green
        (0, 1): "#5a88b0",  # FP - blue
        (1, 0): "#c792d2",  # FN - purple
        (1, 1): "#cf6f2f"   # TP - orange
    }

    # Convert hex to RGB image
    rgb_image = np.zeros((2, 2, 3))

    for i in range(2):
        for j in range(2):
            hex_color = colors[(i, j)]
            rgb_image[i, j] = tuple(int(hex_color[k:k+2], 16)/255 for k in (1, 3, 5))

    ax.imshow(rgb_image)

    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])

    # Add values
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="black"
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUT_DIR)

    rf_csv = find_file_by_keywords(ALLMODELS_DIR, ["rf", "oof"], "RandomForest")
    cnn_csv = find_file_by_keywords(ALLMODELS_DIR, ["cnn", "oof"], "CNN")

    print("Loading files...")
    print("Proposed:", PROPOSED_CSV)
    print("LightGBM:", LIGHTGBM_CSV)
    print("Random Forest:", rf_csv)
    print("CNN:", cnn_csv)

    proposed_df = load_oof(PROPOSED_CSV, "Proposed")
    lightgbm_df = load_oof(LIGHTGBM_CSV, "LightGBM")
    rf_df = load_oof(rf_csv, "RandomForest")
    cnn_df = load_oof(cnn_csv, "CNN")

    print("\nValidating alignment...")
    validate_alignment(proposed_df, lightgbm_df, "Proposed", "LightGBM")
    validate_alignment(proposed_df, rf_df, "Proposed", "RandomForest")
    validate_alignment(proposed_df, cnn_df, "Proposed", "CNN")

    y_true = proposed_df["y_true"].values

    # Simple average fusion from LightGBM + RF + CNN
    simple_avg_prob = (
        lightgbm_df["y_prob"].values +
        rf_df["y_prob"].values +
        cnn_df["y_prob"].values
    ) / 3.0

    models = {
        "Proposed": proposed_df["y_prob"].values,
        "LightGBM": lightgbm_df["y_prob"].values,
        "Random Forest": rf_df["y_prob"].values,
        "CNN": cnn_df["y_prob"].values,
        "Simple Avg": simple_avg_prob
    }

    summary_rows = []
    cms = {}

    for model_name, y_prob in models.items():
        tn, fp, fn, tp = get_confusion_counts(y_true, y_prob, THRESHOLD)
        cms[model_name] = np.array([[tn, fp], [fn, tp]])

        summary_rows.append({
            "Model": model_name,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

        print(f"\n{model_name}")
        print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "confusion_matrix_counts_all_models.csv")
    summary_df.to_csv(summary_csv, index=False)

    # 2x3 layout, final slot hidden
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    ordered_names = ["Proposed", "LightGBM", "Random Forest", "CNN", "Simple Avg"]

    for idx, model_name in enumerate(ordered_names):
        plot_confusion_matrix_panel(axes[idx], cms[model_name], model_name)

    axes[5].axis("off")

    plt.suptitle(f"Pooled Confusion Matrices at Threshold = {THRESHOLD:.2f}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(OUT_DIR, "confusion_matrix_all_models.png")
    out_pdf = os.path.join(OUT_DIR, "confusion_matrix_all_models.pdf")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print("\nSaved files:")
    print(out_png)
    print(out_pdf)
    print(summary_csv)


if __name__ == "__main__":
    main()