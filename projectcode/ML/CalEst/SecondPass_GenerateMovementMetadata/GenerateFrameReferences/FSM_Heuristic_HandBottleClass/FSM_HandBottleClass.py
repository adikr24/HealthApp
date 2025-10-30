#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from difflib import SequenceMatcher

from projectcode.ML.CalEst.SecondPass_GenerateMovementMetadata.GenerateFrameReferences.FSM_Heuristic_HandBottleClass.LabelDetection import FrameOCRRunner

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
CSV_ACTIVITY = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
CSV_ACTIVITY_ALT = BASE / "ActivityFrameSegmentation" / "all_activity_frames.csv"  # optional fallback

FRAMES_ROOT  = Path("/app/mediaFiles/videos/InputVideos/ProteinShakeII/ExtractFrames/ProteinShake")
CSV_OUT = BASE / "bottle_ocr_image.csv"
OUT_CSV = BASE / "fsm_heuristics.csv"

# ---------- Helpers ----------
def _split_classes(s: str):
    return [t.strip().lower() for t in str(s).split("|") if t.strip()]

def _has_all(tokens, required):
    st = set(tokens)
    return all(r in st for r in required)

# ---------- FSM ----------
class FSM:
    def __init__(self):
        pass

    # --- fuzzy helpers ---
    def _fuzzy_match(self, text: str, targets: list[str], threshold: float = 0.7):
        best, best_ratio = None, 0
        for t in targets:
            ratio = SequenceMatcher(None, text, t).ratio()
            if ratio > best_ratio:
                best, best_ratio = t, ratio
        return best if best_ratio >= threshold else None

    def _normalize_ocr_text(self, text: str) -> str:
        """Convert OCR text into known ingredient categories using robust matching."""
        if not text:
            return ""
        import re as _re
        t = text.lower()
        t = _re.sub(r"[^a-z\s]", " ", t)      # drop digits/symbols
        t = _re.sub(r"\s+", " ", t).strip()
        if not t:
            return ""

        yogurt_keys = ["chobani", "oikos", "fage", "greek yogurt", "yoplait", "siggi", "greek"]
        whey_keys   = ["whey", "protein", "gold standard", "optimum", "myprotein", "muscle milk", "isolate", "whey protein"]
        milk_keys   = ["fairlife", "milk", "oat milk", "almond milk", "silk"]
        pb_keys     = ["peanut butter", "peanutbutter", "jif", "skippy", "pb"]

        def contains_any(s: str, keys) -> bool:
            return any(k in s for k in keys)

        # ✅ Correct indentation (these must be OUTSIDE contains_any)
        if contains_any(t, yogurt_keys):
            return "greek_yogurt"
        if contains_any(t, whey_keys):
            return "protein_powder"
        if contains_any(t, milk_keys):
            return "milk"
        if contains_any(t, pb_keys):
            return "peanut_butter"

        # Fallback: token-level fuzzy matching
        tokens = t.split()
        def fuzzy_any(tokens, targets, thr=0.72):
            for tok in tokens:
                for target in targets:
                    if SequenceMatcher(None, tok, target).ratio() >= thr:
                        return True
            return False

        if fuzzy_any(tokens, yogurt_keys, 0.72):
            return "greek_yogurt"
        if fuzzy_any(tokens, whey_keys, 0.72):
            return "protein_powder"
        if fuzzy_any(tokens, milk_keys, 0.72):
            return "milk"
        if fuzzy_any(tokens, pb_keys, 0.72):
            return "peanut_butter"

        # Last resort: return sanitized text (won't trigger early-exit set)
        return t

    # --- main FSM ---
    def run(self):
        # Load activity CSV (merged preferred; fallback optional)
        if CSV_ACTIVITY.exists():
            act = pd.read_csv(CSV_ACTIVITY)
        elif CSV_ACTIVITY_ALT.exists():
            act = pd.read_csv(CSV_ACTIVITY_ALT)
        else:
            raise FileNotFoundError(f"Could not find activity CSV at {CSV_ACTIVITY} or {CSV_ACTIVITY_ALT}")

        act.columns = [c.strip() for c in act.columns]
        all_rows = []

        for _, row in act.iterrows():
            classes = str(row["classes"])
            toks = _split_classes(classes)

            # Trigger only on hand+bottle or hand+scoop
            if not (_has_all(toks, ["hand", "bottle"]) or _has_all(toks, ["hand", "scoop"])):
                continue

            start_idx = int(row["start_idx"])
            lookback, mode = (200, "hand+scoop") if _has_all(toks, ["hand", "scoop"]) else (50, "hand+bottle")
            win_start = max(0, start_idx - lookback)
            win_end = start_idx - 1

            print(f"\n[INFO] State {int(row['state_id'])} | {classes} | Mode={mode} | Frames {win_start}–{win_end}")
            print(f"[INFO] OCR on frames {win_start}–{win_end} in: {FRAMES_ROOT}")

            # Run OCR on the frame window
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
            print(f"[OK] Done OCR on frames {win_start}–{win_end}")

            matched = None  # <-- track if label found
            for r in results:
                fname = r["frame"]
                raw_text = r.get("text", "") or ""
                norm_text = self._normalize_ocr_text(raw_text)

                print(f"  {fname} → {raw_text}  → normalized: {norm_text}")
                all_rows.append({
                    "state_id": int(row["state_id"]),
                    "classes": classes,
                    "frame": fname,
                    "ocr_text": norm_text
                })

                # --- EARLY EXIT if known label detected ---
                if norm_text in {"greek_yogurt", "protein_powder", "milk", "peanut_butter"}:
                    matched = norm_text
                    print(f"[EARLY EXIT] Found '{matched}' → stopping OCR for this state.")
                    break

        print("\n[OK] Finished OCR + fuzzy normalization for hand+bottle and hand+scoop states.")

        # ----- Write OCR output -----
        out_df = pd.DataFrame(all_rows)
        if out_df.empty:
            print("[WARN] No OCR text extracted — nothing to save.")
            return

        if CSV_OUT.exists():
            prev = pd.read_csv(CSV_OUT)
            prev.columns = [c.strip() for c in prev.columns]
            merged = pd.concat([prev, out_df], ignore_index=True)
            merged.drop_duplicates(subset=["state_id", "frame"], keep="last", inplace=True)
            merged.to_csv(CSV_OUT, index=False)
        else:
            out_df.to_csv(CSV_OUT, index=False)

        print(f"\n[OK] OCR results written to: {CSV_OUT}")

# ---------- Run directly ----------
if __name__ == "__main__":
    FSM().run()
