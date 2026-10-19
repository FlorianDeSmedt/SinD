#!/usr/bin/env python3
"""
Rebuild a SIND-style Veh_tracks_meta.csv from Veh_smoothed_tracks.csv.

Columns produced (as in Format.md):
  trackId, initialFrame, finalFrame, Frame_nums, length, width, class,
  CrossType, Signal_Violation_Behavior

Everything except CrossType / Signal_Violation_Behavior is plain aggregation.
Those two need site geometry, supplied via a JSON config (see site_config_example.json):
  arms[name] = {
     "stop_line":      [[x,y],[x,y]]   # motor-vehicle stop line (ground coords, metres)
     "crosswalk_line": [[x,y],[x,y]]   # outer border of the zebra crossing (used by non-motor classes)
     "exit_line":      [[x,y],[x,y]]   # line crossed when LEAVING the intersection on this arm
     "light_column":   "Traffic light 3"   # column in TrafficLight_*.csv that controls this approach
  }
  non_motor_classes: ["bicycle","motorcycle","tricycle"]
  turn_right_exempt: true   # right turns never count as violations (China)

Assumes a right-handed ground frame (y up). If left/right counts look swapped, set
"flip_handedness": true in the config.

Copy this into powershell to run:

python SinD\make_veh_meta.py `
  --tracks "SinD\Data\Tianjin\8_2_1\Veh_smoothed_tracks.csv" `
  --lights "SinD\Data\Tianjin\8_2_1\TrafficLight_8_2_1.csv" `
  --config "SinD\tianjin_config.json" `
  --out "SinD\my_Veh_tracks_meta.csv"
"""
import argparse, json
import numpy as np
import pandas as pd


def seg_cross(p0, p1, a, b):
    """Parameter t in [0,1] along p0->p1 where it crosses segment a-b, else None."""
    d, e, w = p1 - p0, b - a, a - p0
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-12:
        return None
    t = (w[0] * e[1] - w[1] * e[0]) / den
    u = (w[0] * d[1] - w[1] * d[0]) / den
    return t if (0 <= t <= 1 and 0 <= u <= 1) else None


def first_cross(P, T, line, t_min=-np.inf):
    """Earliest time the path P (timestamps T) crosses the polyline `line` after t_min."""
    pts = [np.asarray(q, float) for q in line]
    best = None
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        for i in range(len(P) - 1):
            t = seg_cross(P[i], P[i + 1], a, b)
            if t is not None:
                tm = T[i] + t * (T[i + 1] - T[i])
                if tm > t_min:
                    if best is None or tm < best:
                        best = tm
                    break  # first hit on this segment is its earliest
    return best


def light_state(lights, col, t_ms):
    """State of `col` at time t (last change event at or before t). 0=red 1=green 3=yellow."""
    i = np.searchsorted(lights["ts"].values, t_ms, side="right") - 1
    return np.nan if i < 0 else lights[col].iloc[i]


def classify_track(g, cfg, lights):
    g = g.sort_values("frame_id")
    cls = g["agent_type"].iloc[0]
    non_motor = cls in cfg["non_motor_classes"]
    T = g["timestamp_ms"].values.astype(float)
    L = g["length"].fillna(0).values
    # "front wheels" ~ front-centre of the bounding box
    # reference point along the long axis: fraction of full length from the centre
    # (0.5 = front bumper, 0 = box centre; wheels are somewhere in between). cfg key: ref_point_frac
    f = cfg.get("ref_point_frac", 0.5)
    P = np.c_[g["x"].values + f * L * np.cos(g["yaw_rad"].values),
              g["y"].values + f * L * np.sin(g["yaw_rad"].values)]
    arms = cfg["arms"]
    names = cfg["arm_order_ccw"]

    # entry: earliest crossing of any approach line
    entry = {}
    for n, a in arms.items():
        line = a["crosswalk_line"] if (non_motor and "crosswalk_line" in a) else a["stop_line"]
        t = first_cross(P, T, line)
        if t is not None:
            entry[n] = t
    if not entry:
        return "Others", "No violation of traffic lights"
    en = min(entry, key=entry.get)
    t_in = entry[en]

    # exit: earliest exit-line crossing after entry, on a different arm
    ex = {}
    for n, a in arms.items():
        if n == en:
            continue
        t = first_cross(P, T, a["exit_line"], t_min=t_in)
        if t is not None:
            ex[n] = t
    if not ex:
        return "Others", "No violation of traffic lights"
    xn = min(ex, key=ex.get)

    k = (names.index(xn) - names.index(en)) % len(names)
    if cfg.get("flip_handedness"):
        k = (-k) % len(names)
    cross = {1: "RightTurn", 2: "StraightCross", 3: "LeftTurn"}.get(k, "Others")

    if cross in ("LeftTurn", "StraightCross"):
        s = light_state(lights, arms[en]["light_column"], t_in)
        if s == 0:
            return cross, "red-light running"
        if s == 3:
            return cross, "yellow-light running"
    return cross, "No violation of traffic lights"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--lights", required=True, help="TrafficLight_<record>.csv")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="Veh_tracks_meta.csv")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    df = pd.read_csv(a.tracks)
    lights = pd.read_csv(a.lights)
    lights = lights.rename(columns={"timestamp(ms)": "ts"}).sort_values("ts").reset_index(drop=True)

    rows = []
    for tid, g in df.groupby("track_id"):
        cross, viol = classify_track(g, cfg, lights)
        rows.append(dict(
            trackId=tid,
            initialFrame=g["frame_id"].min(),
            finalFrame=g["frame_id"].max(),
            Frame_nums=g["frame_id"].max() - g["frame_id"].min() + 1,
            length=g["length"].median(),   # SIND doesn't say mean vs median; check against their files
            width=g["width"].median(),
            **{"class": g["agent_type"].iloc[0]},
            CrossType=cross,
            Signal_Violation_Behavior=viol,
        ))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} tracks -> {a.out}")


if __name__ == "__main__":
    main()