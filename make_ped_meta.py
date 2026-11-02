#!/usr/bin/env python3
r"""
Build a pedestrian meta file (like Ped_tracks_meta.csv) and add crosswalk / signal columns.

Columns kept from the original:  trackId, initialFrame, finalFrame, Frame_nums, class
Columns added:
  Crosswalk                  crosswalk strip(s) the pedestrian walked along, in order, e.g. "N" or "N;E"  ("None" if none)
  Entry_frames               frame in which the track stepped onto each of those crosswalks
  Light_at_entry             pedestrian-light state at that moment (red / yellow / green / unknown)
  Signal_Violation_Behavior  "red-light running" | "yellow-light running" | "No violation of traffic lights"

Definition (same idea as for vehicles at the stop line):
  * a pedestrian "crosses" a crosswalk if the track is observed stepping from outside onto the zebra polygon and then
    covers at least --min-span of the strip's length inside it;
  * the violation is judged by the light state at the moment of stepping on (a light turning red while still
    on the crosswalk is not a violation);
  * tracks that already start on the crosswalk have unknown entry state -> no violation is flagged.

Config: needs a "crosswalks" block ({name: {"inside": polyline, "outside": polyline, "light_column": "Traffic light n"}}),
see tianjin_config.json.  Light codes in the light file: 0 red, 1 green, 3 yellow.

  python make_ped_meta.py --tracks Ped_smoothed_tracks.csv --lights TrafficLight_8_2_1.csv --config tianjin_config.json
  python make_ped_meta.py ... --check          # table of light states at entry for every candidate light column
  python make_ped_meta.py ... --auto-light     # per crosswalk, use the light column that is green most often at entry
"""
import argparse, json
import numpy as np
import pandas as pd
from matplotlib.path import Path

NAME = {0: "red", 1: "green", 3: "yellow"}



def strip_polygon(cw):
    """Non-self-intersecting polygon of a crosswalk strip from its two long borders, whatever their point order.
    Both borders are sorted along the strip's main axis and clipped to the stretch where BOTH exist.
    Returns (polygon Nx2, centre, axis, s_lo, s_hi)."""
    ins = np.array(cw["inside"], float); out = np.array(cw["outside"], float)
    allp = np.vstack([ins, out]); ctr = allp.mean(0)
    axis = np.linalg.svd(allp - ctr)[2][0]
    def prep(p):
        s = (p - ctr) @ axis; o = np.argsort(s); return p[o], s[o]
    ins, s_in = prep(ins); out, s_out = prep(out)
    lo, hi = max(s_in[0], s_out[0]), min(s_in[-1], s_out[-1])
    def clip(p, s):
        keep = (s > lo) & (s < hi)
        x0 = np.interp([lo, hi], s, p[:, 0]); y0 = np.interp([lo, hi], s, p[:, 1])
        return np.vstack([[x0[0], y0[0]], p[keep], [x0[1], y0[1]]])
    poly = np.vstack([clip(ins, s_in), clip(out, s_out)[::-1]])
    return poly, ctr, axis, lo, hi

def build_polys(cfg):
    out = {}
    for n, c in cfg["crosswalks"].items():
        poly, ctr, ax, lo, hi = strip_polygon(c)
        out[n] = dict(path=Path(poly), centre=ctr, axis=ax, smin=lo, smax=hi, col=c["light_column"])
    return out


def state_at(lights, col, t_ms):
    i = np.searchsorted(lights["ts"].values, t_ms, side="right") - 1
    return np.nan if i < 0 else lights[col].iloc[i]


def crossings(g, polys, min_span):
    """List of (crosswalk, entry_frame, entry_time_ms) in order of entry for one pedestrian track."""
    g = g.sort_values("frame_id")
    P = g[["x", "y"]].values; fr = g.frame_id.values; ts = g.timestamp_ms.values
    ev = []
    for n, p in polys.items():
        ins = p["path"].contains_points(P)
        if ins.sum() < 3:
            continue
        s = (P[ins] - p["centre"]) @ p["axis"]
        if (s.max() - s.min()) < min_span * (p["smax"] - p["smin"]):
            continue
        i0 = int(np.argmax(ins))
        entry_known = i0 > 0 and not ins[i0 - 1]          # observed stepping from outside onto the zebra
        ev.append((n, int(fr[i0]), float(ts[i0]), entry_known))
    return sorted(ev, key=lambda e: e[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True); ap.add_argument("--lights", required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--out", default="my_Ped_tracks_meta.csv")
    ap.add_argument("--min-span", type=float, default=0.6, help="fraction of the strip length that must be covered")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--auto-light", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.config)); polys = build_polys(cfg)
    df = pd.read_csv(a.tracks, usecols=["track_id", "frame_id", "timestamp_ms", "agent_type", "x", "y"])
    lights = pd.read_csv(a.lights).rename(columns={"timestamp(ms)": "ts"}).sort_values("ts").reset_index(drop=True)
    lcols = [c for c in lights.columns if c.startswith("Traffic light")]

    events = {}
    for tid, g in df.groupby("track_id"):
        events[tid] = crossings(g, polys, a.min_span)

    if a.check or a.auto_light:
        print("light state at entry, per crosswalk and candidate light column (r/y/g = red/yellow/green counts)")
        for n in polys:
            ts_ = [e[2] for ev in events.values() for e in ev if e[0] == n and e[3]]
            if not ts_:
                print(f"  {n}: no observed entries"); continue
            rows = []
            for c in lcols:
                st = [state_at(lights, c, t) for t in ts_]
                r, y, gr = st.count(0), st.count(3), st.count(1)
                rows.append((c, r, y, gr))
                print(f"  {n} {c:16s} r={r:3d} y={y:3d} g={gr:3d}   green share {gr/len(ts_):.2f}" + ("   <- configured" if c == polys[n]['col'] else ""))
            if a.auto_light:
                best = max(rows, key=lambda r: r[3])[0]
                print(f"  -> {n}: using {best}"); polys[n]["col"] = best

    rows = []
    for tid, g in df.groupby("track_id"):
        ev = events[tid]
        worst, parts = 0, []
        for n, f, t, known in ev:
            s = state_at(lights, polys[n]["col"], t) if known else np.nan
            parts.append((n, f, NAME.get(s, "unknown")))
            if s == 0: worst = max(worst, 2)
            elif s == 3: worst = max(worst, 1)
        rows.append(dict(
            trackId=tid, initialFrame=g.frame_id.min(), finalFrame=g.frame_id.max(),
            Frame_nums=g.frame_id.max() - g.frame_id.min() + 1, **{"class": g.agent_type.iloc[0]},
            Crosswalk=";".join(p[0] for p in parts) or "None",
            Entry_frames=";".join(str(p[1]) for p in parts),
            Light_at_entry=";".join(p[2] for p in parts),
            Signal_Violation_Behavior=["No violation of traffic lights", "yellow-light running", "red-light running"][worst]))
    out = pd.DataFrame(rows); out.to_csv(a.out, index=False)
    print(f"\nwrote {len(out)} pedestrians -> {a.out}")
    print(out.Signal_Violation_Behavior.value_counts().to_string())
    print("on a crosswalk:", int((out.Crosswalk != "None").sum()), "| not on any crosswalk:", int((out.Crosswalk == "None").sum()))


if __name__ == "__main__":
    main()
