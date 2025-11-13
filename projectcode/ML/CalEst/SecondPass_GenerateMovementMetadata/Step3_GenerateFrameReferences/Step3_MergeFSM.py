#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# ---------- CONFIG ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/FSM")
EDIBLE_CSV     = BASE / "fsm_heuristics_EdibleClass.csv"
HANDBOTTLE_CSV = BASE / "fsm_heuristics_HandBottle.csv"
HANDONLY_CSV   = BASE / "fsm_heuristics_HandOnly.csv"
OUT_CSV        = BASE / "fsm_heuristics_Merged.csv"

def load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    df = pd.read_csv(p, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def norm_str(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.replace({"": "NA", "nan": "NA", "NaN": "NA", "None": "NA"})

def main():
    # ---- Load all CSVs ----
    df_ed = load_csv(EDIBLE_CSV)
    df_hb = load_csv(HANDBOTTLE_CSV)
    df_ho = load_csv(HANDONLY_CSV)

    # ---- Ensure columns exist ----
    for df in [df_ed, df_hb, df_ho]:
        if "OOI" not in df.columns:
            df["OOI"] = "NA"
        if "bottle_ocr" not in df.columns:
            df["bottle_ocr"] = "NA"
        if "VOI" not in df.columns:
            df["VOI"] = "NA"
        df["OOI"] = norm_str(df["OOI"])
        df["bottle_ocr"] = norm_str(df["bottle_ocr"])
        df["VOI"] = norm_str(df["VOI"])

    # ---- Step 1: Merge Edible → HandBottle ----
    ed_key = df_ed[["state_id", "OOI"]].rename(columns={"OOI": "OOI_ed"})
    merged = ed_key.merge(df_hb, on="state_id", how="left")

    # Override OOI only when Edible has a meaningful value
    edible_has_ooi = (
        merged["OOI_ed"].notna()
        & (merged["OOI_ed"].astype(str).str.strip() != "")
        & (merged["OOI_ed"].str.upper() != "NA")
    )
    merged["OOI"] = norm_str(merged["OOI"])
    merged.loc[edible_has_ooi, "OOI"] = merged.loc[edible_has_ooi, "OOI_ed"]
    merged.drop(columns=["OOI_ed"], inplace=True)

    # ---- Step 2: Bring in HandOnly rows (non-NA OOI only) ----
    ho_non_na = df_ho.loc[
        (df_ho["OOI"].notna())
        & (df_ho["OOI"].astype(str).str.strip() != "")
        & (df_ho["OOI"].str.upper() != "NA")
    ].copy()

    # We only care about unique states where OOI exists (e.g., almond, walnut)
    ho_non_na = ho_non_na[["state_id", "OOI"]].rename(columns={"OOI": "OOI_ho"})

    # Merge into main (override OOI where HandOnly provides a value)
    merged = merged.merge(ho_non_na, on="state_id", how="left")
    handonly_has_ooi = (
        merged["OOI_ho"].notna()
        & (merged["OOI_ho"].astype(str).str.strip() != "")
        & (merged["OOI_ho"].str.upper() != "NA")
    )
    merged.loc[handonly_has_ooi, "OOI"] = merged.loc[handonly_has_ooi, "OOI_ho"]
    merged.drop(columns=["OOI_ho"], inplace=True)

    # ---- Normalize again just in case ----
    for col in ["OOI", "bottle_ocr", "VOI"]:
        merged[col] = norm_str(merged[col])

    # ---- Save final merged ----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)
    print(f"[DONE] Final merged CSV written → {OUT_CSV}")

if __name__ == "__main__":
    main()
