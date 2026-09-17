import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = "EchoNet-Dynamic/metadata.csv"
OUTPUT_CSV = "EchoNet-Dynamic/qualitative_captions.csv"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_columns = ["FileName", "EF", "EDV", "ESV"]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# EF classification
# ============================================================
# Main indicator of LV systolic function.
#
# EF >= 55      -> preserved
# 40 <= EF < 55 -> mildly reduced
# EF < 40       -> reduced
# ============================================================

def classify_ef(ef):

    if pd.isna(ef):
        return "unknown"

    if ef >= 55:
        return "preserved"

    elif ef >= 40:
        return "mildly reduced"

    else:
        return "reduced"


def classify_contraction(ef):

    if pd.isna(ef):
        return "variable"

    if ef >= 55:
        return "marked"

    elif ef >= 40:
        return "moderate"

    else:
        return "limited"


def classify_emptying(ef):

    if pd.isna(ef):
        return "variable"

    if ef >= 55:
        return "effective"

    elif ef >= 40:
        return "moderately effective"

    else:
        return "limited"


def classify_pumping(ef):

    if pd.isna(ef):
        return "variable"

    if ef >= 55:
        return "good"

    elif ef >= 40:
        return "moderate"

    else:
        return "reduced"


# ============================================================
# EDV classification
# ============================================================
# EDV is interpreted relative to the dataset distribution.
#
# < 33rd percentile -> relatively limited filling
# 33rd-67th         -> moderate filling
# >= 67th           -> substantial filling
# ============================================================

edv_q33 = df["EDV"].quantile(0.33)
edv_q67 = df["EDV"].quantile(0.67)


def classify_filling(edv):

    if pd.isna(edv):
        return "variable"

    if edv < edv_q33:
        return "relatively limited"

    elif edv < edv_q67:
        return "moderate"

    else:
        return "substantial"


# ============================================================
# ESV classification
# ============================================================
# Instead of using absolute ESV, use ESV/EDV.
#
# ESV/EDV < 0.35 -> small residual cavity
# 0.35-0.60      -> moderate residual cavity
# >= 0.60        -> substantial residual cavity
#
# This is more meaningful because ESV depends strongly on EDV.
# ============================================================

def classify_residual_volume(edv, esv):

    if pd.isna(edv) or pd.isna(esv) or edv <= 0:
        return "variable"

    ratio = esv / edv

    if ratio < 0.35:
        return "small"

    elif ratio < 0.60:
        return "moderate"

    else:
        return "substantial"


# ============================================================
# Generate fixed-format caption
# ============================================================

def generate_caption(row):

    ef = row["EF"]
    edv = row["EDV"]
    esv = row["ESV"]

    filling = classify_filling(edv)
    contraction = classify_contraction(ef)
    residual = classify_residual_volume(edv, esv)
    emptying = classify_emptying(ef)
    systolic_function = classify_ef(ef)
    pumping = classify_pumping(ef)

    caption = (
        f"The left ventricle shows {filling} filling during diastole, "
        f"followed by a {contraction} reduction in ventricular cavity size "
        f"during systole. The cavity becomes {residual} after contraction, "
        f"indicating {emptying} ventricular emptying. Overall, the observed "
        f"ventricular motion is consistent with {systolic_function} "
        f"left ventricular systolic function and {pumping} pumping performance."
    )

    return caption


# ============================================================
# Generate captions
# ============================================================

df["caption"] = df.apply(generate_caption, axis=1)


# ============================================================
# Save
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved captions to: {OUTPUT_CSV}")

print("\nExample annotations:\n")

print(
    df[
        ["FileName", "EF", "EDV", "ESV", "caption"]
    ].head(10).to_string(index=False)
)
