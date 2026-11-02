#!/usr/bin/env python3
r"""
Replay a SIND record on the aerial image.

 * vehicles    = rotated boxes (x, y, yaw_rad, length, width), coloured by class, with a short trail and the trackId
 * pedestrians = circles with a trail and the id "P<trackId>"   (--peds Ped_smoothed_tracks.csv)
 * stop lines change colour with the live state of the vehicle light that controls them (red / green / yellow)
 * crosswalk outlines (if the config has a "crosswalks" block) change colour with the live pedestrian-light state
 * --meta / --ped-meta: tracks flagged in the meta files get a thick outline
        red outline = red-light running, orange outline = yellow-light running
   (use the authors' files, or the ones made by make_veh_meta.py / make_ped_meta.py)
 * --focus <vehicle trackId> or --focus-ped <pedestrian trackId>  replays around that track and highlights it

Ground -> image pixel:  u = k*(U + S*x) ,  v = k*(V - S*y)
  S, U, V are fitted on a REFERENCE image that is 2576 px wide (the Tianjin.jpg copy used for calibration).
  k = (actual image width) / (reference width), so any resolution of the same frame (e.g. the 3840 px original) works.
  Override with --scale/--origin/--ref-width if you calibrated on another image.

Examples
  python replay_tracks.py --image tianjin_clean_aerial.jpg --config tianjin_config.json \
      --tracks Data/Tianjin/8_2_1/Veh_smoothed_tracks.csv --lights Data/Tianjin/8_2_1/TrafficLight_8_2_1.csv \
      --peds Data/Tianjin/8_2_1/Ped_smoothed_tracks.csv \
      --meta my_Veh_tracks_meta.csv --ped-meta my_Ped_tracks_meta.csv --start 0 --end 600 --out replay.mp4

  python replay_tracks.py ... --focus 123 --out track123.gif
  python replay_tracks.py ... --focus-ped 57 --out ped57.mp4


  $d = "SinD\Data\Tianjin\8_2_1"

python SinD\make_ped_meta.py --tracks $d\Ped_smoothed_tracks.csv --lights $d\TrafficLight_8_2_1.csv --config SinD\tianjin_config.json --check


To run with pedestrians:
python SinD\replay_tracks.py --image SinD\tianjin_clean_aerial.jpg --config SinD\tianjin_config.json `
  --tracks $d\Veh_smoothed_tracks.csv --peds $d\Ped_smoothed_tracks.csv --lights $d\TrafficLight_8_2_1.csv `
  --meta SinD\my_Veh_tracks_meta.csv --ped-meta SinD\my_Ped_tracks_meta.csv --start 0 --end 12005 --out SinD\replay.mp4
"""
import argparse, json, shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.image import imread
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D

CLASS_COL = {"car": "#2ecc71", "truck": "#9b59b6", "bus": "#f1c40f", "bicycle": "#3498db",
             "motorcycle": "#e67e22", "tricycle": "#1abc9c"}
LIGHT_COL = {0: "#e74c3c", 1: "#2ecc71", 3: "#f1c40f"}   # SIND: 0 red, 1 green, 3 yellow
VIOL_COL = {"red-light running": "#ff0000", "yellow-light running": "#ff8c00"}


def corners(x, y, yaw, L, W):
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([L / 2, L / 2, -L / 2, -L / 2]); dy = np.array([W / 2, -W / 2, -W / 2, W / 2])
    return np.c_[x + dx * c - dy * s, y + dx * s + dy * c]


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


def label(ax, x, y, s, fs=9):
    return ax.text(x, y, s, color="white", fontsize=fs, fontweight="bold", ha="center", va="center", zorder=8,
                   path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tracks", required=True, help="Veh_smoothed_tracks.csv")
    ap.add_argument("--lights", required=True)
    ap.add_argument("--peds", help="Ped_smoothed_tracks.csv")
    ap.add_argument("--meta", help="Veh_tracks_meta.csv (original or rebuilt)")
    ap.add_argument("--ped-meta", help="Ped meta with Signal_Violation_Behavior (make_ped_meta.py output)")
    ap.add_argument("--scale", type=float, default=35.05, help="pixels per metre, on the reference image")
    ap.add_argument("--origin", type=float, nargs=2, default=[725.7, 1322.0], help="pixel (u v) of ground (0,0), on the reference image")
    ap.add_argument("--ref-width", type=float, default=2576.0, help="width in px of the image S and origin refer to")
    ap.add_argument("--start", type=int); ap.add_argument("--end", type=int)
    ap.add_argument("--focus", type=int, help="vehicle trackId to highlight; window defaults to its lifetime +-3 s")
    ap.add_argument("--focus-ped", type=int, help="pedestrian trackId to highlight; window defaults to its lifetime +-3 s")
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--fps", type=float, default=10, help="output fps; 10 with --step 1 = real time")
    ap.add_argument("--no-labels", action="store_true", help="do not print the trackIds")
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
    pdf = pd.read_csv(a.peds, usecols=["track_id", "frame_id", "timestamp_ms", "x", "y"]) if a.peds else \
        pd.DataFrame(columns=["track_id", "frame_id", "timestamp_ms", "x", "y"])
    lights = pd.read_csv(a.lights).rename(columns={"timestamp(ms)": "ts"}).sort_values("ts").reset_index(drop=True)
    meta = pd.read_csv(a.meta).set_index("trackId") if a.meta else None
    pmeta = pd.read_csv(a.ped_meta).set_index("trackId") if a.ped_meta else None

    allf = pd.concat([df.frame_id, pdf.frame_id])
    f0, f1 = (a.start if a.start is not None else allf.min()), (a.end if a.end is not None else allf.max())
    for tid, src, name in ((a.focus, df, "vehicle"), (a.focus_ped, pdf, "pedestrian")):
        if tid is None: continue
        g = src[src.track_id == tid]
        if g.empty: raise SystemExit(f"{name} track {tid} not found")
        if a.start is None: f0 = int(g.frame_id.min()) - 30
        if a.end is None: f1 = int(g.frame_id.max()) + 30
    win = df[(df.frame_id >= f0) & (df.frame_id <= f1)]
    pwin = pdf[(pdf.frame_id >= f0) & (pdf.frame_id <= f1)]
    frames = np.sort(pd.concat([win.frame_id, pwin.frame_id]).unique())[::a.step]
    if len(frames) == 0: raise SystemExit("no frames in the selected window")
    by_frame = {f: g for f, g in win.groupby("frame_id")}
    pby_frame = {f: g for f, g in pwin.groupby("frame_id")}
    ftime = {}
    for f, g in pby_frame.items(): ftime[f] = float(g.timestamp_ms.iloc[0])
    for f, g in by_frame.items(): ftime[f] = float(g.timestamp_ms.iloc[0])

    def paths_of(d):
        out = {}
        for tid, g in d.sort_values("frame_id").groupby("track_id"):
            out[tid] = (g.frame_id.values, tp(g[["x", "y"]].values))
        return out
    paths, ppaths = paths_of(win), paths_of(pwin)

    fig = plt.figure(figsize=(13, 13 * h / w), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)

    stop = {}
    for arm, d in cfg["arms"].items():
        p = tp(np.array(d["stop_line"]))
        stop[arm] = (ax.plot(p[:, 0], p[:, 1], lw=6, solid_capstyle="butt", color="gray", zorder=3)[0], d["light_column"])
    cwalk = {}
    for name, c in cfg.get("crosswalks", {}).items():
        poly = tp(strip_polygon(c)[0])
        pa = Polygon(poly, closed=True, fill=False, lw=2.5, ec="gray", ls="-", alpha=0.9, zorder=2)
        ax.add_patch(pa); cwalk[name] = (pa, c["light_column"])
    info = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", color="white", fontsize=12, fontweight="bold",
                   bbox=dict(fc="black", alpha=0.55, lw=0))
    legend = [Line2D([0], [0], marker="s", ls="", color=c, label=n) for n, c in CLASS_COL.items()]
    if a.peds:
        legend.append(Line2D([0], [0], marker="o", ls="", mfc="white", mec="black", label="pedestrian"))
    if meta is not None or pmeta is not None:
        legend += [Line2D([0], [0], marker="s", ls="", mfc="none", mec=c, mew=2, label=n) for n, c in VIOL_COL.items()]
    if cwalk:
        legend.append(Line2D([0], [0], color="gray", lw=2.5, label="crosswalk (colour = ped. light)"))
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.6)

    polys, trails, texts, shown = {}, {}, {}, set()
    pcirc, ptrails, ptexts, pshown = {}, {}, {}, set()
    PR = 0.45 * S      # pedestrian circle radius in px

    def light_state(col, t):
        i = np.searchsorted(lights["ts"].values, t, side="right") - 1
        return None if i < 0 else int(lights[col].iloc[i])

    def trail_data(pathd, tid, f):
        fr, uv = pathd[tid]
        j1 = np.searchsorted(fr, f, side="right"); j0 = max(0, j1 - a.trail)
        return uv[j0:j1, 0], uv[j0:j1, 1]

    def update(i):
        f = frames[i]
        t = ftime[f]
        now = set()
        g = by_frame.get(f)
        if g is not None:
            for r in g.itertuples(index=False):
                tid = r.track_id; now.add(tid)
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
                if a.trail: trails[tid].set_data(*trail_data(paths, tid, f))
                if not a.no_labels:
                    cx, cy = tp(np.array([[r.x, r.y]]))[0]
                    if tid not in texts: texts[tid] = label(ax, cx, cy, str(tid))
                    texts[tid].set_position((cx, cy - 0.9 * S - 6)); texts[tid].set_visible(True)
        for tid in shown - now:
            polys[tid].set_visible(False); trails[tid].set_data([], [])
            if tid in texts: texts[tid].set_visible(False)
        shown.clear(); shown.update(now)

        pnow = set()
        pg = pby_frame.get(f)
        if pg is not None:
            for r in pg.itertuples(index=False):
                tid = r.track_id; pnow.add(tid)
                cx, cy = tp(np.array([[r.x, r.y]]))[0]
                if tid not in pcirc:
                    ec, lw = "black", 1.0
                    if pmeta is not None and tid in pmeta.index:
                        v = pmeta.loc[tid, "Signal_Violation_Behavior"]
                        if v in VIOL_COL: ec, lw = VIOL_COL[v], 3
                    if tid == a.focus_ped: ec, lw = "cyan", 3.5
                    pcirc[tid] = Circle((cx, cy), PR, fc="white", ec=ec, lw=lw, alpha=0.95, zorder=7)
                    ax.add_patch(pcirc[tid])
                    ptrails[tid] = ax.plot([], [], lw=1.5, color="white", alpha=0.8, zorder=4)[0]
                pcirc[tid].center = (cx, cy); pcirc[tid].set_visible(True)
                if a.trail: ptrails[tid].set_data(*trail_data(ppaths, tid, f))
                if not a.no_labels:
                    if tid not in ptexts: ptexts[tid] = label(ax, cx, cy, f"P{tid}", fs=8)
                    ptexts[tid].set_position((cx, cy - PR - 9)); ptexts[tid].set_visible(True)
        for tid in pshown - pnow:
            pcirc[tid].set_visible(False); ptrails[tid].set_data([], [])
            if tid in ptexts: ptexts[tid].set_visible(False)
        pshown.clear(); pshown.update(pnow)

        for arm, (line, col) in stop.items():
            line.set_color(LIGHT_COL.get(light_state(col, t), "gray"))
        for name, (pa, col) in cwalk.items():
            pa.set_edgecolor(LIGHT_COL.get(light_state(col, t), "gray"))
        msg = f"frame {f}   t = {t / 1000:6.1f} s   vehicles: {len(now)}"
        if a.peds: msg += f"   pedestrians: {len(pnow)}"
        info.set_text(msg)
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
