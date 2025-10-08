#!/usr/bin/env python3
# audit_min.py  — quick YOLO label sanity check (COCO+custom ok)
import yaml, argparse, os
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def load_yaml(p): 
    with open(p,"r") as f: return yaml.safe_load(f)

def lbl_path(img: Path):
    parts = list(img.parts)
    # find the LAST occurrence of 'images' in the path
    idxs = [i for i, p in enumerate(parts) if p == "images"]
    if idxs:
        idx = idxs[-1]
        parts[idx] = "labels"
        lp = Path(*parts)
        return lp.with_suffix(".txt")
    # fallback: no 'images' segment found
    s = str(img)
    i = s.rfind("/images/")
    if i != -1:
        s = s[:i] + "/labels/" + s[i+len("/images/"):]
    return Path(os.path.splitext(s)[0] + ".txt")

def parse_line(ln):
    t = ln.split()
    if len(t)!=5: return None
    try:
        cid = int(float(t[0])); x,y,w,h = map(float, t[1:])
        ok_box = 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
        return cid, ok_box
    except: return None

def audit(data_yaml):
    cfg = load_yaml(data_yaml); names = cfg["names"]
    if isinstance(names, dict): names = [names[k] for k in sorted(names,key=lambda k:int(k))]
    n = len(names)
    stats = {"imgs":0,"missing":0,"bad_id":0,"bad_box":0}
    for split in ("train","val"):
        items = cfg[split]; items = [items] if isinstance(items,str) else items
        for root in items:
            for p in Path(root).rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    stats["imgs"] += 1
                    lp = lbl_path(p)
                    if not lp.exists(): stats["missing"] += 1; continue
                    with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln: continue
                            parsed = parse_line(ln)
                            if not parsed: stats["bad_id"] += 1; continue
                            cid, ok_box = parsed
                            if not (0 <= cid < n): stats["bad_id"] += 1
                            elif not ok_box:       stats["bad_box"] += 1
    return n, stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    args = ap.parse_args()
    n, s = audit(args.data)
    print(f"[audit] classes={n} images={s['imgs']} missing_labels={s['missing']} bad_id_lines={s['bad_id']} bad_bbox_lines={s['bad_box']}")
    print("PASS ✅" if s["missing"]==s["bad_id"]==s["bad_box"]==0 else "CHECK ⚠️")
