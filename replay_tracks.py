#!/usr/bin/env python3
"""
Replay a SIND record on the aerial image.

 * vehicles = rotated boxes (from x, y, yaw_rad, length, width), coloured by class, with a short trail
 * pedestrians (optional) = dots
 * stop lines change colour with the live signal state (red / green / yellow) of the light that controls them
 * with --meta, vehicles flagged in Veh_tracks_meta.csv get a thick outline
        red outline = red-light running, orange outline = yellow-light running
   (use the authors' meta, or the one made by make_veh_meta.py, to eyeball how the labelling behaves)
 * --focus <trackId> replays around one vehicle and highlights it

Ground -> image pixel:  u = k*(U + S*x) ,  v = k*(V - S*y)
  S, U, V are fitted on a REFERENCE image that is 2576 px wide (the Tianjin.jpg copy used for calibration).
  k = (actual image width) / (reference width), so any resolution of the same frame (e.g. the 3840 px original) works.
  Override with --scale/--origin/--ref-width if you calibrated on another image.

Examples
  python replay_tracks.py --image Tianjin.jpg --config tianjin_config.json \
      --tracks Data/Tianjin/8_2_1/Veh_smoothed_tracks.csv --lights Data/Tianjin/8_2_1/TrafficLight_8_2_1.csv \
      --meta Data/Tianjin/8_2_1/Veh_tracks_meta.csv --start 0 --end 600 --out replay.mp4

  python replay_tracks.py ... --focus 123 --out track123.gif

  To run:
  python SinD\replay_tracks.py `
  --image SinD\Data\Tianjin\Tianjin.jpg `
  --config SinD\tianjin_config.json `
  --tracks SinD\Data\Tianjin\8_2_1\Veh_smoothed_tracks.csv `
  --lights SinD\Data\Tianjin\8_2_1\TrafficLight_8_2_1.csv `
  --peds SinD\Data\Tianjin\8_2_1\Ped_smoothed_tracks.csv `
  --meta SinD\my_Veh_tracks_meta.csv `
  --start 0 --end 600 --out SinD\replay.mp4
"""
import argparse, json, shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

CLASS_COL = {"car": "#2ecc71", "truck": "#9b59b6", "bus": "#f1c40f", "bicycle": "#3498db",
             "motorcycle": "#e67e22", "tricycle": "#1abc9c"}
LIGHT_COL = {0: "#e74c3c", 1: "#2ecc71", 3: "#f1c40f"}   # SIND: 0 red, 1 green, 3 yellow
VIOL_COL = {"red-light running": "#ff0000", "yellow-light running": "#ff8c00"}


def corners(x, y, yaw, L, W):
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([L / 2, L / 2, -L / 2, -L / 2]); dy = np.array([W / 2, -W / 2, -W / 2, W / 2])
    return np.c_[x + dx * c - dy * s, y + dx * s + dy * c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--lights", required=True)
    ap.add_argument("--peds")
    ap.add_argument("--meta")
    ap.add_argument("--scale", type=float, default=35.05, help="pixels per metre, on the reference image")
    ap.add_argument("--origin", type=float, nargs=2, default=[725.7, 1322.0], help="pixel (u v) of ground (0,0), on the reference image")
    ap.add_argument("--ref-width", type=float, default=2576.0, help="width in px of the image S and origin refer to")
    ap.add_argument("--start", type=int); ap.add_argument("--end", type=int)
    ap.add_argument("--focus", type=int, help="trackId to highlight; window defaults to its lifetime +-3 s")
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--fps", type=float, default=10, help="output fps; 10 with --step 1 = real time")
    ap.add_argument("--no-labels", action="store_true", help="do not print the trackId next to each vehicle")
    ap.add_argument("--trail", type=int, default=15, help="trail length in frames (0 = off)")
    ap.add_argument("--out", default="replay.mp4")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    img = imread(a.image)
    h, w = img.shape[:2]
    k = w / a.ref_width                      # resolution factor between the calibration image and this one
    S, (U, V) = a.scale * k, (a.origin[0] * k, a.origin[1] * k)
    print(f"image {w}x{h}: resolution factor {k:.3f} -> {S:.2f} px/m, origin ({U:.1f}, {V:.1f})")
    tp = lambda xy: np.c_[U + S * xy[:, 0], V - S * xy[:, 1]]

    cols = ["track_id", "frame_id", "timestamp_ms", "agent_type", "x", "y", "yaw_rad", "length", "width"]
    df = pd.read_csv(a.tracks, usecols=cols)
    lights = pd.read_csv(a.lights).rename(columns={"timestamp(ms)": "ts"}).sort_values("ts").reset_index(drop=True)
    meta = pd.read_csv(a.meta).set_index("trackId") if a.meta else None

    f0, f1 = (a.start if a.start is not None else df.frame_id.min()), (a.end if a.end is not None else df.frame_id.max())
    if a.focus is not None:
        g = df[df.track_id == a.focus]
        if g.empty:
            raise SystemExit(f"track {a.focus} not found")
        if a.start is None: f0 = int(g.frame_id.min()) - 30
        if a.end is None: f1 = int(g.frame_id.max()) + 30
    win = df[(df.frame_id >= f0) & (df.frame_id <= f1)]
    frames = np.sort(win.frame_id.unique())[::a.step]
    by_frame = {f: g for f, g in win.groupby("frame_id")}
    # per-track pixel paths for trails
    paths = {}
    for tid, g in win.sort_values("frame_id").groupby("track_id"):
        paths[tid] = (g.frame_id.values, tp(g[["x", "y"]].values))
    ped_by_frame = {}
    if a.peds:
        pdf = pd.read_csv(a.peds, usecols=["track_id", "frame_id", "x", "y"])
        pdf = pdf[(pdf.frame_id >= f0) & (pdf.frame_id <= f1)]
        ped_by_frame = {f: tp(g[["x", "y"]].values) for f, g in pdf.groupby("frame_id")}

    fig = plt.figure(figsize=(13, 13 * h / w), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)

    stop = {}
    for arm, d in cfg["arms"].items():
        p = tp(np.array(d["stop_line"]))
        stop[arm] = (ax.plot(p[:, 0], p[:, 1], lw=6, solid_capstyle="butt", color="gray", zorder=3)[0], d["light_column"])
    info = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", color="white", fontsize=12, fontweight="bold",
                   bbox=dict(fc="black", alpha=0.55, lw=0))
    peds_art = ax.scatter([], [], s=14, c="white", edgecolors="black", linewidths=0.4, zorder=5)
    legend = [Line2D([0], [0], marker="s", ls="", color=c, label=k) for k, c in CLASS_COL.items()]
    if meta is not None:
        legend += [Line2D([0], [0], marker="s", ls="", mfc="none", mec=c, mew=2, label=k) for k, c in VIOL_COL.items()]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.6)

    polys, trails, texts, shown = {}, {}, {}, set()

    def light_state(col, t):
        i = np.searchsorted(lights["ts"].values, t, side="right") - 1
        return None if i < 0 else int(lights[col].iloc[i])

    def update(k):
        f = frames[k]
        g = by_frame[f]
        t = float(g.timestamp_ms.iloc[0])
        now = set()
        for r in g.itertuples(index=False):
            tid = r.track_id
            now.add(tid)
            L = 1.8 if np.isnan(r.length) else r.length
            W = 0.7 if np.isnan(r.width) else r.width
            xy = tp(corners(r.x, r.y, r.yaw_rad, L, W))
            if tid not in polys:
                ec, lw = "black", 0.6
                if meta is not None and tid in meta.index:
                    v = meta.loc[tid, "Signal_Violation_Behavior"]
                    if v in VIOL_COL: ec, lw = VIOL_COL[v], 3
                if tid == a.focus: ec, lw = "cyan", 3.5
                polys[tid] = Polygon(xy, closed=True, fc=CLASS_COL.get(r.agent_type, "#bdc3c7"), ec=ec, lw=lw, alpha=0.9, zorder=6)
                ax.add_patch(polys[tid])
                trails[tid] = ax.plot([], [], lw=1.5, color=CLASS_COL.get(r.agent_type, "#bdc3c7"), alpha=0.8, zorder=4)[0]
            polys[tid].set_xy(xy); polys[tid].set_visible(True)
            if not a.no_labels:
                cx, cy = tp(np.array([[r.x, r.y]]))[0]
                if tid not in texts:
                    texts[tid] = ax.text(cx, cy, str(tid), color="white", fontsize=9, fontweight="bold", ha="center", va="center",
                                         zorder=8, path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
                texts[tid].set_position((cx, cy - 0.9 * S - 6)); texts[tid].set_visible(True)
            if a.trail:
                fr, uv = paths[tid]
                j1 = np.searchsorted(fr, f, side="right"); j0 = max(0, j1 - a.trail)
                trails[tid].set_data(uv[j0:j1, 0], uv[j0:j1, 1])
        for tid in shown - now:
            polys[tid].set_visible(False); trails[tid].set_data([], [])
            if tid in texts: texts[tid].set_visible(False)
        shown.clear(); shown.update(now)
        for arm, (line, col) in stop.items():
            s = light_state(col, t)
            line.set_color(LIGHT_COL.get(s, "gray"))
        peds_art.set_offsets(ped_by_frame.get(f, np.empty((0, 2))))
        info.set_text(f"frame {f}   t = {t / 1000:6.1f} s   vehicles: {len(g)}")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = a.out
    if out.lower().endswith(".gif"):
        writer = PillowWriter(fps=a.fps)
    else:
        if not shutil.which("ffmpeg"):
            try:   # `pip install imageio-ffmpeg` ships a private ffmpeg binary
                import imageio_ffmpeg
                matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                print("No ffmpeg found (try: pip install imageio-ffmpeg) -> writing a GIF instead")
                out = out.rsplit(".", 1)[0] + ".gif"
        writer = PillowWriter(fps=a.fps) if out.endswith(".gif") else FFMpegWriter(fps=a.fps, bitrate=4000)
    print(f"rendering {len(frames)} frames (frame {frames[0]}..{frames[-1]}) -> {out}")
    anim.save(out, writer=writer, progress_callback=lambda i, n: print(f"  {i}/{n}", end="\r") if i % 50 == 0 else None)
    print("\nwrote", out)


if __name__ == "__main__":
    main()