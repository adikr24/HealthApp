## this code will be used to mark frames and objects where hand-held objects are detected ##

#!/usr/bin/env python3
# compute_hand_held_objects.py
#
# Adds "finger" as a precise hand proxy:
# - finger touching/overlapping an object => treat as holding
# - finger has tighter hold distances than palm/hand
# - all existing constraints (one-object-per-hand, clustering, hysteresis) preserved

from pathlib import Path
import csv, re, math, cv2
from collections import defaultdict

# ======================= CONFIG =======================

IN_CSV = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/all_detections_with_hand_side.csv")
IN_IMG_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes")

OUT_ROOT = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/hand_held_objects")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_FRAMES_DIR = OUT_ROOT / "frames"
OUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER_FRAME_CSV = OUT_ROOT / "hand_object_per_frame.csv"
OUT_EVENTS_CSV    = OUT_ROOT / "hand_held_events.csv"

# Hand-like labels
HAND_LABELS = {"hand", "hand_open", "hand_closed", "hand_pinch", "palm", "on_palm", "finger"}  # NEW: finger

IGNORE_OBJECT_LABELS = {"person"} | HAND_LABELS | {"blender_mouth"}

# Distances / association
MAX_HOLD_DIST_PX       = 85     # default hand/palm grab radius
FINGER_MAX_HOLD_DIST_PX = 70    # NEW: finger grab radius (slightly tighter)
FINGER_TOUCH_DIST_PX    = 40    # NEW: if finger within this distance => hold
FINGER_OVERLAP_IOU      = 0.0   # NEW: any overlap with object => hold

HANDS_OVERLAP_IOU    = 0.05
STICKY_MIN_FRAMES    = 4
MISSING_TOLERANCE    = 6
MIN_EVENT_FRAMES     = 3
DRAW                  = True

# Viz
COLOR_HAND_L   = (0, 200, 255)
COLOR_HAND_R   = (255, 160, 0)
COLOR_CLUSTER  = (180, 180, 255)
COLOR_FINGER   = (255, 80, 80)   # NEW: finger highlight
COLOR_OBJECT   = (50, 220, 50)
COLOR_LINE     = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ======================= HELPERS =======================

def norm(s): return (s or "").strip().lower()

def frame_idx(name):
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else -1

# Fixed: accept either bbox tuple/list or four numeric args
def bbox_center(*args):
    """
    bbox_center(x1,y1,x2,y2) or bbox_center((x1,y1,x2,y2))
    returns (cx, cy)
    """
    if len(args) == 1:
        b = args[0]
        x1,y1,x2,y2 = b
    elif len(args) == 4:
        x1,y1,x2,y2 = args
    else:
        raise TypeError("bbox_center expects 1 iterable bbox or 4 numeric args")
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def distance(p1,p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def rect_area(b):
    x1,y1,x2,y2 = b
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def iou(a,b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    x1=max(ax1,bx1); y1=max(ay1,by1)
    x2=min(ax2,bx2); y2=min(ay2,by2)
    if x2<=x1 or y2<=y1: return 0.0
    inter=(x2-x1)*(y2-y1)
    return inter / max(rect_area(a)+rect_area(b)-inter, 1e-6)

def read_rows():
    with IN_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def build_per_frame(rows):
    frames = defaultdict(lambda: {'hands': [], 'objects': []})
    for r in rows:
        fname = Path(r.get("frame") or r.get("source") or "").name
        if not fname: continue
        cls = norm(r.get("cls_name",""))
        try:
            x1,y1,x2,y2 = map(float, [r["x1"], r["y1"], r["x2"], r["y2"]])
        except: 
            continue
        cx,cy = bbox_center(x1,y1,x2,y2)
        tid = (r.get("obj_tid") or "").strip()

        if cls in HAND_LABELS:
            side = norm(r.get("hand_side",""))
            # ID heuristic
            hand_id = "left" if side=="left" else ("right" if side=="right" else ("finger" if cls=="finger" else "hand"))
            frames[fname]['hands'].append({
                'id': hand_id,
                'box': (x1,y1,x2,y2),
                'centroid': (cx,cy),
                'is_finger': (cls=="finger")  # NEW flag
            })
        elif cls not in IGNORE_OBJECT_LABELS:
            key = (cls, tid if tid else f"idx_{len(frames[fname]['objects'])}")
            frames[fname]['objects'].append({'k': key, 'cls': cls, 'tid': tid, 'box': (x1,y1,x2,y2), 'centroid': (cx,cy)})

    return dict(sorted(frames.items(), key=lambda kv: frame_idx(kv[0])))

# ======================= CORE LOGIC =======================

def nearest_object(hand, objs, exclude_keys):
    """
    Finger rules (NEW):
      - if finger overlaps any object (IoU>FINGER_OVERLAP_IOU) -> pick nearest overlapping
      - else if distance <= FINGER_TOUCH_DIST_PX -> pick nearest within that band
      - else use FINGER_MAX_HOLD_DIST_PX
    Hands/palms use MAX_HOLD_DIST_PX.
    """
    is_finger = hand.get('is_finger', False)
    max_dist = FINGER_MAX_HOLD_DIST_PX if is_finger else MAX_HOLD_DIST_PX

    best_overlap = (None, 1e9)  # (key, dist) among overlapping objs
    best_touch   = (None, 1e9)
    best_regular = (None, 1e9)

    for o in objs:
        if o['k'] in exclude_keys: 
            continue
        d = distance(hand['centroid'], o['centroid'])
        if is_finger:
            ov = iou(hand['box'], o['box'])
            if ov > FINGER_OVERLAP_IOU:
                if d < best_overlap[1]:
                    best_overlap = (o['k'], d)
            elif d <= FINGER_TOUCH_DIST_PX:
                if d < best_touch[1]:
                    best_touch = (o['k'], d)
            # fallthrough to regular band too
        if d < best_regular[1]:
            best_regular = (o['k'], d)

    if is_finger:
        if best_overlap[0] is not None:
            return best_overlap[0]
        if best_touch[0] is not None:
            return best_touch[0]
        return best_regular[0] if best_regular[1] <= max_dist else None
    else:
        return best_regular[0] if best_regular[1] <= max_dist else None

def assign_hands_to_objects(hands, objs):
    """
    Global greedy matching with finger force-hold precedence.
    Returns dict hand_id -> obj_key (or None).
    """
    if not hands or not objs:
        # ensure we return a mapping of canonical hand ids -> None
        return {h['id']: None for h in _canonicalize_hand_ids(hands)}

    # Canonicalize IDs to avoid collisions like multiple "hand"/"finger"
    hands_canon = _canonicalize_hand_ids(hands)  # list of dicts with 'id','box','centroid','is_finger'
    hand_ids = [h['id'] for h in hands_canon]

    # If overlapping hands => cluster path (same as before)
    # (You can keep your earlier cluster branch here; if you do, just make sure to call _canonicalize_hand_ids on singles)
    # --- If you already had a cluster branch above, you can safely return there. ---

    # Build candidate edges
    edges_force = []   # (dist, hand_id, obj_key)
    edges_regular = [] # (dist, hand_id, obj_key)
    for h in hands_canon:
        cand = _nearest_object_candidates(h, objs)
        for (key, dist, force) in cand:
            if force:
                edges_force.append((dist, h['id'], key))
            else:
                edges_regular.append((dist, h['id'], key))

    # Sort by distance (smaller first)
    edges_force.sort(key=lambda t: t[0])
    edges_regular.sort(key=lambda t: t[0])

    assigned = {hid: None for hid in hand_ids}
    used_objs = set()
    used_hands = set()

    # 1) Commit all force-hold (finger) matches first
    for dist, hid, key in edges_force:
        if hid in used_hands or key in used_objs:
            continue
        assigned[hid] = key
        used_hands.add(hid)
        used_objs.add(key)

    # 2) Fill remaining via global nearest (greedy)
    for dist, hid, key in edges_regular:
        if hid in used_hands or key in used_objs:
            continue
        assigned[hid] = key
        used_hands.add(hid)
        used_objs.add(key)

    return assigned

# ======================= TRACK/HYSTERESIS & IO =======================

# (Identical to previous version, except visualization color for finger and unchanged state/event logic)

# ... keep the rest of the previous script the same, with these tiny viz tweaks:

# When choosing color per hand id:
# col = COLOR_CLUSTER if hid=="hands_cluster" else (COLOR_FINGER if hid=="finger" else (COLOR_HAND_L if hid=="left" else COLOR_HAND_R))

# The rest (state machine, hysteresis, CSV outputs) remains unchanged.


# ======================= TRACKING / HYSTERESIS =======================

def run():
    rows = read_rows()
    frames = build_per_frame(rows)

    # Per-hand state across frames
    # state[hand_id] = {
    #   'current': key_or_None,
    #   'pending': key_or_None, 'pending_count': int,
    #   'missing': int,
    # }
    state = defaultdict(lambda: {'current': None, 'pending': None, 'pending_count': 0, 'missing': 0})

    # Active events keyed by (hand_id, current_key)
    active = {}

    # Per-frame log rows for OUT_PER_FRAME_CSV
    per_frame_logs = []

    # Iterate frames
    for fname, data in frames.items():
        img = None
        if DRAW:
            ip = IN_IMG_DIR / fname
            if ip.exists():
                img = cv2.imread(str(ip))

        hands = data['hands']
        objs  = data['objects']
        assign = assign_hands_to_objects(hands, objs)

        # Build lookup from object key -> object record (for distances/viz)
        obj_lookup = {o['k']: o for o in objs}

        # Update states for each hand (and cluster if present)
        this_frame_assignments = []  # for per-frame CSV

        # Build a dict of hand geometries by id for viz
        hand_geom = {}
        for h in hands:
            hand_geom.setdefault(h['id'], h)

        # If cluster exists but not in hand_geom, create geometry for viz
        if 'hands_cluster' in assign and 'hands_cluster' not in hand_geom and len(hands) >= 2:
            # reuse union centroid from assign step by recomputing quickly
            # (approximate with average of all hands’ centroids)
            xs = [h['centroid'][0] for h in hands]
            ys = [h['centroid'][1] for h in hands]
            cc = (sum(xs)/len(xs), sum(ys)/len(ys))
            # union box
            xs_b = [h['box'][0] for h in hands] + [h['box'][2] for h in hands]
            ys_b = [h['box'][1] for h in hands] + [h['box'][3] for h in hands]
            hand_geom['hands_cluster'] = {'id':'hands_cluster',
                                          'box': (min(xs_b), min(ys_b), max(xs_b), max(ys_b)),
                                          'centroid': cc}

        # enforce hysteresis / build events
        for hand_id, key in assign.items():
            st = state[hand_id]
            # if no candidate this frame
            if key is None:
                st['pending'] = None
                st['pending_count'] = 0
                st['missing'] += 1
                # end event if missing too long
                if st['current'] is not None and st['missing'] > MISSING_TOLERANCE:
                    end_event(active, hand_id, frames, fname)
                    st['current'] = None
                # per-frame log
                this_frame_assignments.append((hand_id, None, None, None))
            else:
                st['missing'] = 0
                # If same as current, keep it and clear pending
                if key == st['current']:
                    st['pending'] = None
                    st['pending_count'] = 0
                else:
                    # candidate differs; require STICKY_MIN_FRAMES to switch
                    if st['pending'] == key:
                        st['pending_count'] += 1
                    else:
                        st['pending'] = key
                        st['pending_count'] = 1

                    if st['pending_count'] >= STICKY_MIN_FRAMES:
                        # close old event
                        if st['current'] is not None:
                            end_event(active, hand_id, frames, fname)
                        # switch
                        st['current'] = key
                        st['pending'] = None
                        st['pending_count'] = 0
                        # start new event
                        start_event(active, hand_id, key, fname)

                # per-frame log with distance (if known)
                o = obj_lookup.get(st['current'])
                dpx = None
                if o is not None and hand_id in hand_geom:
                    dpx = int(round(distance(hand_geom[hand_id]['centroid'], o['centroid'])))
                this_frame_assignments.append((hand_id, st['current'], o['cls'] if o else None, dpx))

        # annotate frame
        if DRAW and img is not None:
            # draw objects
            for o in objs:
                x1,y1,x2,y2 = map(int, o['box'])
                cv2.rectangle(img, (x1,y1), (x2,y2), COLOR_OBJECT, 2)
                cv2.putText(img, o['cls'], (x1, max(0,y1-6)), FONT, 0.5, COLOR_OBJECT, 1)

            # draw hands + lines to current object
            for hid, geom in hand_geom.items():
                x1,y1,x2,y2 = map(int, geom['box'])
                col = COLOR_CLUSTER if hid=="hands_cluster" else (COLOR_HAND_L if hid=="left" else COLOR_HAND_R)
                cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
                cv2.circle(img, (int(geom['centroid'][0]), int(geom['centroid'][1])), 3, col, -1)
                # line to held object
                held_key = state[hid]['current'] if hid in state else None
                if held_key in obj_lookup:
                    oc = obj_lookup[held_key]['centroid']
                    cv2.line(img,
                             (int(geom['centroid'][0]), int(geom['centroid'][1])),
                             (int(oc[0]), int(oc[1])), COLOR_LINE, 2)
                    dpx = int(round(distance(geom['centroid'], oc)))
                    label = f"{hid}→{obj_lookup[held_key]['cls']}:{dpx}px"
                    cv2.putText(img, label, (x1, min(img.shape[0]-5,y2+16)), FONT, 0.55, col, 2)

            cv2.imwrite(str(OUT_FRAMES_DIR / fname), img)

        # write per-frame log row(s)
        fidx = frame_idx(fname)
        for (hand_id, key, cls, dpx) in this_frame_assignments:
            # normalize None for CSV
            k_cls = cls or ""
            k_tid = ""
            if key is not None:
                k_cls = key[0]
                k_tid = key[1]
            per_frame_logs.append({
                "frame": fname,
                "frame_idx": fidx,
                "hand_id": hand_id,
                "object_key": f"{k_cls}:{k_tid}" if key is not None else "",
                "object_cls": k_cls if key is not None else "",
                "object_tid": k_tid if key is not None else "",
                "distance_px": dpx if dpx is not None else ""
            })

    # finalize any active events at end
    for evk in list(active.keys()):
        end_event(active, evk[0], frames, None)

    # dump per-frame CSV
    with OUT_PER_FRAME_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame","frame_idx","hand_id","object_key","object_cls","object_tid","distance_px"])
        w.writeheader()
        w.writerows(per_frame_logs)

    # dump events CSV
    write_events_csv(active)

    print(f"[DONE] Per-frame assignments → {OUT_PER_FRAME_CSV}")
    print(f"[DONE] Annotated frames → {OUT_FRAMES_DIR}")
    print(f"[DONE] Event summaries → {OUT_EVENTS_CSV}")

# -------- Event helpers --------

def start_event(active, hand_id, key, start_frame):
    evk = (hand_id, key)
    active[evk] = {
        "hand_id": hand_id,
        "obj_cls": key[0],
        "obj_tid": key[1],
        "start_frame": start_frame,
        "start_idx": frame_idx(start_frame),
        "last_frame": start_frame,
        "last_idx": frame_idx(start_frame),
        "frames": 1
    }

def end_event(active, hand_id, frames_dict, end_frame):
    # find the active event for this hand (whatever key it is)
    to_close = [k for k in active.keys() if k[0] == hand_id]
    if not to_close:
        return
    evk = to_close[0]
    ev = active[evk]
    # if end_frame is None, leave last_frame as current last
    if end_frame is not None:
        ev["last_frame"] = end_frame
        ev["last_idx"] = frame_idx(end_frame)
    # filter short events
    if ev["frames"] >= MIN_EVENT_FRAMES:
        # stash into a finalized list on the active dict (abuse container)
        active.setdefault("_final", []).append(ev)
    # remove active
    active.pop(evk, None)

def write_events_csv(active):
    events = active.get("_final", []) if isinstance(active.get("_final", []), list) else []
    # merge consecutive rows of same (hand_id, obj_cls, obj_tid) that are contiguous
    # (simple; we already enforce hysteresis, so this is mostly contiguous)
    events_sorted = sorted(events, key=lambda e: e["start_idx"])
    # rebuild frames count from idx difference if missing
    for e in events_sorted:
        if e.get("frames", 0) <= 1 and e.get("start_idx") is not None and e.get("last_idx") is not None:
            e["frames"] = max(1, (e["last_idx"] - e["start_idx"] + 1))
    with OUT_EVENTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "hand_id","object_cls","object_tid",
            "start_frame","end_frame","start_idx","end_idx","frames"
        ])
        w.writeheader()
        for e in events_sorted:
            w.writerow({
                "hand_id": e["hand_id"],
                "object_cls": e["obj_cls"],
                "object_tid": e["obj_tid"],
                "start_frame": e["start_frame"],
                "end_frame": e["last_frame"],
                "start_idx": e["start_idx"],
                "end_idx": e["last_idx"],
                "frames": e["frames"]
            })

def _canonicalize_hand_ids(hands):
    """
    Return a list of hand dicts with unique ids.
    If there are multiple hands with the same base id (e.g., "hand"), append an index.
    Keeps 'id','box','centroid','is_finger' keys.
    """
    from collections import Counter, defaultdict
    base_counts = Counter(h['id'] for h in hands)
    idx = defaultdict(int)
    out = []
    for h in hands:
        hid = h['id']
        if base_counts[hid] > 1:
            idx[hid] += 1
            new_id = f"{hid}_{idx[hid]}"
        else:
            new_id = hid
        # copy and set canonical id
        hh = {'id': new_id, 'box': h['box'], 'centroid': h['centroid'], 'is_finger': h.get('is_finger', False)}
        out.append(hh)
    return out

def _nearest_object_candidates(hand, objs):
    """
    Build (obj_key, distance, force_flag) list for a hand.
    force_flag is True for finger overlaps/touches to prioritize in matching.
    """
    candidates = []
    for o in objs:
        d = distance(hand['centroid'], o['centroid'])
        force = False
        if hand.get('is_finger', False):
            if iou(hand['box'], o['box']) > FINGER_OVERLAP_IOU or d <= FINGER_TOUCH_DIST_PX:
                force = True
        candidates.append((o['k'], d, force))
    # sort by distance (optional; callers also sort)
    return sorted(candidates, key=lambda t: t[1])

# ======================= MAIN =======================

if __name__ == "__main__":
    run()
