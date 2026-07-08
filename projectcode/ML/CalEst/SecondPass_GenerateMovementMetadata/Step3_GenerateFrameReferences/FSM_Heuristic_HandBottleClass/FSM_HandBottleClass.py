#!/usr/bin/env python3
"""
Purpose: Hand-bottle FSM heuristic for generating frame-level and state-level
object-of-interest metadata.
This script inspects activity segments involving hand+bottle or hand+scoop,
uses OCR on nearby frames to read bottle text, normalizes the detected labels,
and writes/merges the inferred OOI and bottle_ocr values into CSV outputs.
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from difflib import SequenceMatcher

from projectcode.ML.CalEst.SecondPass_GenerateMovementMetadata.Step3_GenerateFrameReferences.FSM_Heuristic_HandBottleClass.LabelDetection import FrameOCRRunner

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
CSV_ACTIVITY = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
CSV_ACTIVITY_ALT = BASE / "ActivityFrameSegmentation" / "all_activity_frames.csv"

FRAMES_ROOT  = Path("/app/mediaFiles/videos/InputVideos/ProteinShakeIII/ExtractFrames/ProteinShake")
CSV_OUT = BASE / "bottle_ocr_image.csv"                      # frame-level log
OUT_CSV = BASE / "FSM" / "fsm_heuristics_HandBottle.csv"     # state-level heuristics

EARLY_SET = {"protein_powder", "greek_yogurt"}               # keep as-is per your last note

# ---------- Helpers ----------
def _split_classes(s: str):
    return [t.strip().lower() for t in str(s).split("|") if t.strip()]

def _has_all(tokens, required):
    st = set(tokens)
    return all(r in st for r in required)

def _ensure_cols_after_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ... classes, OOI, bottle_ocr, ... order."""
    if "classes" not in df.columns:
        return df
    cols = list(df.columns)
    for c in ["OOI", "bottle_ocr"]:
        if c in cols:
            cols.remove(c)
    insert_pos = cols.index("classes") + 1
    cols[insert_pos:insert_pos] = ["OOI", "bottle_ocr"]
    return df.reindex(columns=cols)

def _sanitize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Central keyword sets (used for both direct and fuzzy matches)
YOGURT_KEYS = ["chobani", "oikos", "fage", "greek yogurt", "yoplait", "siggi", "greek"]
WHEY_KEYS   = ["whey", "protein", "gold standard", "optimum", "myprotein", "muscle milk", "isolate", "whey protein"]
MILK_KEYS   = ["fairlife", "milk", "oat milk", "almond milk", "silk"]
PB_KEYS     = ["peanut butter", "peanutbutter", "jif", "skippy", "pb"]

def _normalize_with_token(raw_text: str) -> tuple[str, str]:
    """
    Return (normalized_label, matched_token).
    - normalized_label ∈ {greek_yogurt, protein_powder, milk, peanut_butter} or "".
    - matched_token is the exact keyword that fired (e.g., 'chobani', 'whey').
    """
    if not raw_text:
        return "", ""
    t = _sanitize(raw_text)
    if not t:
        return "", ""

    def contains_any_get(s: str, keys):
        for k in keys:
            if k in s:
                return k
        return None

    # Direct contains
    if (k := contains_any_get(t, YOGURT_KEYS)):
        return "greek_yogurt", k
    if (k := contains_any_get(t, WHEY_KEYS)):
        return "protein_powder", k
    if (k := contains_any_get(t, MILK_KEYS)):
        return "milk", k
    if (k := contains_any_get(t, PB_KEYS)):
        return "peanut_butter", k

    # Fuzzy token-level fallback
    tokens = t.split()
    def fuzzy_token_match(tokens, targets, thr=0.72):
        best = ("", 0.0)
        for tok in tokens:
            for target in targets:
                r = SequenceMatcher(None, tok, target).ratio()
                if r >= thr and r > best[1]:
                    best = (target, r)
        return best[0] if best[1] > 0 else ""

    if (k := fuzzy_token_match(tokens, YOGURT_KEYS)):
        return "greek_yogurt", k
    if (k := fuzzy_token_match(tokens, WHEY_KEYS)):
        return "protein_powder", k
    if (k := fuzzy_token_match(tokens, MILK_KEYS)):
        return "milk", k
    if (k := fuzzy_token_match(tokens, PB_KEYS)):
        return "peanut_butter", k

    return "", ""  # no match

# ---------- FSM ----------
class FSM:
    def run(self):
        # Load activity CSV
        if CSV_ACTIVITY.exists():
            act = pd.read_csv(CSV_ACTIVITY)
        elif CSV_ACTIVITY_ALT.exists():
            act = pd.read_csv(CSV_ACTIVITY_ALT)
        else:
            raise FileNotFoundError(f"Missing activity CSV at {CSV_ACTIVITY} or {CSV_ACTIVITY_ALT}")
        act.columns = [c.strip() for c in act.columns]

        frame_log_rows = []   # per-frame OCR log
        early_norm = {}       # state_id -> normalized label
        early_token = {}      # state_id -> matched keyword token (for bottle_ocr)

        for _, row in act.iterrows():
            classes = str(row["classes"])
            toks = _split_classes(classes)
            if not (_has_all(toks, ["hand", "bottle"]) or _has_all(toks, ["hand", "scoop"])):
                continue

            state_id = int(row["state_id"])
            start_idx = int(row["start_idx"])
            lookback = 200 if _has_all(toks, ["hand", "scoop"]) else 50
            win_start = max(0, start_idx - lookback)
            win_end = start_idx - 1

            ocr_runner = FrameOCRRunner(
                frames_dir=FRAMES_ROOT,
                start_idx=win_start,
                end_idx=win_end,
                langs=['en'],
                use_gpu=True,
                do_grayscale=True,
                do_upscale=True,
                target_short_side=1024,
            )
            results = ocr_runner.run(print_each=False)

            found = False
            for r in results:
                fname = r["frame"]
                raw_text = r.get("text", "") or ""
                norm, token = _normalize_with_token(raw_text)

                # frame-level log stores the normalized text we saw this frame
                frame_log_rows.append({
                    "state_id": state_id,
                    "classes": classes,
                    "frame": fname,
                    "ocr_text": norm if norm else _sanitize(raw_text)
                })

                if not found and norm in EARLY_SET:
                    early_norm[state_id] = norm        # OOI
                    early_token[state_id] = token      # bottle_ocr (e.g., 'chobani' or 'whey')
                    found = True
                    break  # early-exit once matched

        # ----- Write per-frame OCR log -----
        if frame_log_rows:
            out_df = pd.DataFrame(frame_log_rows)
            if CSV_OUT.exists():
                prev = pd.read_csv(CSV_OUT)
                prev.columns = [c.strip() for c in prev.columns]
                merged = pd.concat([prev, out_df], ignore_index=True)
                merged.drop_duplicates(subset=["state_id", "frame"], keep="last", inplace=True)
                merged.to_csv(CSV_OUT, index=False)
            else:
                CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(CSV_OUT, index=False)

        # ----- Build state-level OOI + bottle_ocr and write/merge -----
        add_rows = []
        for _, r in act.iterrows():
            sid = int(r["state_id"])
            ooi = early_norm.get(sid, "NA")
            boc = early_token.get(sid, "NA")  # keyword token (e.g., 'chobani', 'whey'); empty if none
            add_rows.append({"state_id": sid, "OOI": ooi, "bottle_ocr": boc})
        add_df = pd.DataFrame(add_rows)

        base = act.copy()
        base.columns = [c.strip() for c in base.columns]

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        if OUT_CSV.exists():
            existing = pd.read_csv(OUT_CSV)
            existing.columns = [c.strip() for c in existing.columns]
            merged = existing.merge(add_df, on="state_id", how="left", suffixes=("", "_new"))
            for c in ["OOI", "bottle_ocr"]:
                nc = f"{c}_new"
                if nc in merged.columns:
                    if c in merged.columns:
                        merged[c] = merged[nc].where(merged[nc].notna(), merged[c])
                    else:
                        merged[c] = merged[nc]
                    merged.drop(columns=[nc], inplace=True)
            merged = _ensure_cols_after_classes(merged)
            merged.to_csv(OUT_CSV, index=False)
        else:
            if "OOI" in base.columns: base.drop(columns=["OOI"], inplace=True)
            if "bottle_ocr" in base.columns: base.drop(columns=["bottle_ocr"], inplace=True)
            out_fsm = base.merge(add_df, on="state_id", how="left")
            out_fsm = _ensure_cols_after_classes(out_fsm)
            out_fsm.to_csv(OUT_CSV, index=False)

if __name__ == "__main__":
    FSM().run()
    