"""
Time-to-collision (TTC) for vehicle and pedestrian tracks.

Shapes
  * Vehicles    : oriented rectangles (x, y = centre; yaw_rad; length x width).
  * Pedestrians : circles (radius PED_RADIUS).

Method (constant velocity, 2D, no rotation): both agents keep their CURRENT velocity.
TTC = first time t >= 0 at which the shapes touch. Already touching -> TTC = 0 ('overlap').
  * vehicle - vehicle    : swept separating-axis test on the 4 edge normals (exact for rectangles).
  * vehicle - pedestrian : exact. In the vehicle's frame the pedestrian's centre is a moving point and
                           the vehicle is a rectangle grown by PED_RADIUS (a rounded rectangle) =
                           2 boxes + 4 corner circles; TTC is the earliest entry into any of them.
  * pedestrian - pedestrian (optional): circle-circle.

'gap' = distance between the shapes in metres (negative = overlapping).
  vehicle-pedestrian: exact. vehicle-vehicle: largest separation over the 4 axes (a lower bound).

Long-stationary vehicles (parked etc.) can be excluded (EXCLUDE_LONG_STATIONARY). A vehicle is
"stationary" while its speed is below STOP_SPEED; a continuous stationary period longer than
STATIONARY_MAX_S is treated as parked and the WHOLE period is dropped (as agent and as partner).
Vehicles that wait at a red light (shorter stops) stay in.

Outputs (OUT_DIR): ttc_per_agent.csv, ttc_per_pair.csv, ttc_frames.csv, excluded_stationary.csv
"""
import numpy as np
import pandas as pd

from pathlib import Path

# ----------------------------------------------------------------------------- settings


BASE = Path(__file__).resolve().parent          # the folder containing compute_ttc.py (SinD)
PED_FILE = BASE / "Data" / "Tianjin" / "8_2_1" / "Ped_smoothed_tracks.csv"
VEH_FILE = BASE / "Data" / "Tianjin" / "8_2_1" / "Veh_smoothed_tracks.csv"
OUT_DIR = BASE / "outputs_ttc"
OUT_DIR.mkdir(exist_ok=True)                    # the script doesn't create it by itself

PED_RADIUS = 0.25          # m, pedestrian modelled as a circle
MAX_DIST = 30.0            # m, ignore pairs whose centres are further apart than this
TTC_HORIZON = 10.0         # s, TTC above this is treated as "no conflict"
TTC_FRAME_EXPORT = 5.0     # s, frame-level rows with TTC below this are saved
THRESHOLDS = (1.0, 1.5, 3.0)   # s, for "time spent below threshold" metrics
INCLUDE_PED_PED = False    # pedestrian-pedestrian conflicts (usually not of interest)
MIN_CLOSING_SPEED = 0.0    # m/s, optionally ignore very slow closing (noise), e.g. 0.2
FRAME_DT = 0.1001          # s (timestamp_ms diff is 100.1 ms)

EXCLUDE_LONG_STATIONARY = True   # <- switch for the stationary-vehicle exclusion
STOP_SPEED = 0.3                 # m/s, below this a vehicle counts as stationary
STATIONARY_MAX_S = 120.0         # s, stationary periods LONGER than this count as parked


# ----------------------------------------------------------------------------- loading
def load():
    veh = pd.read_csv(VEH_FILE)
    ped = pd.read_csv(PED_FILE)

    veh["id"] = "V" + veh["track_id"].astype(str)
    veh["kind"] = "vehicle"
    veh["theta"] = veh["yaw_rad"]            # yaw_rad is consistent with v_lon (heading_rad is not)

    ped["id"] = "P" + ped["track_id"].astype(str).str.replace("^P", "", regex=True)
    ped["kind"] = "pedestrian"
    ped["theta"] = 0.0
    ped["length"] = 2 * PED_RADIUS
    ped["width"] = 2 * PED_RADIUS

    cols = ["id", "kind", "agent_type", "frame_id", "timestamp_ms", "x", "y", "vx", "vy",
            "theta", "length", "width"]
    return veh[cols].copy(), ped[cols].copy()


def flag_long_stationary(veh):
    """Boolean Series (aligned with veh.index): frame belongs to a stationary period > STATIONARY_MAX_S."""
    v = veh.sort_values(["id", "frame_id"])
    slow = np.hypot(v["vx"], v["vy"]) < STOP_SPEED
    run = (slow != slow.groupby(v["id"]).shift()).groupby(v["id"]).cumsum()
    run_len_s = slow.groupby([v["id"], run]).transform("size") * FRAME_DT
    return (slow & (run_len_s > STATIONARY_MAX_S)).reindex(veh.index)


# ----------------------------------------------------------------------------- pairs
def build_pairs(veh, ped):
    """All simultaneous pairs per frame, split by type (a = vehicle, b = vehicle/pedestrian)."""
    def merge(a, b, same):
        m = a.merge(b, on="frame_id", suffixes=("_a", "_b"))
        if same:
            m = m[m["id_a"] < m["id_b"]]
        d = np.hypot(m["x_a"] - m["x_b"], m["y_a"] - m["y_b"])
        return m[d < MAX_DIST].reset_index(drop=True)

    vv = merge(veh, veh, True)
    vp = merge(veh, ped, False)
    pp = merge(ped, ped, True) if INCLUDE_PED_PED else None
    return vv, vp, pp


# ----------------------------------------------------------------------------- TTC core
def ttc_rect_rect(p):
    """Swept separating-axis test, oriented rectangles. Returns ttc (inf if none), gap."""
    xa, ya = p["x_a"].to_numpy(), p["y_a"].to_numpy()
    xb, yb = p["x_b"].to_numpy(), p["y_b"].to_numpy()
    th_a, th_b = p["theta_a"].to_numpy(), p["theta_b"].to_numpy()
    ha_l, ha_w = p["length_a"].to_numpy() / 2, p["width_a"].to_numpy() / 2
    hb_l, hb_w = p["length_b"].to_numpy() / 2, p["width_b"].to_numpy() / 2
    dx, dy = xb - xa, yb - ya
    vx = p["vx_b"].to_numpy() - p["vx_a"].to_numpy()
    vy = p["vy_b"].to_numpy() - p["vy_a"].to_numpy()

    ua = np.stack([np.cos(th_a), np.sin(th_a)], 1); va = np.stack([-np.sin(th_a), np.cos(th_a)], 1)
    ub = np.stack([np.cos(th_b), np.sin(th_b)], 1); vb = np.stack([-np.sin(th_b), np.cos(th_b)], 1)

    t_enter = np.full(len(p), -np.inf)
    t_exit = np.full(len(p), np.inf)
    gap = np.full(len(p), -np.inf)
    for n in (ua, va, ub, vb):
        nx, ny = n[:, 0], n[:, 1]
        ra = ha_l * np.abs(nx * ua[:, 0] + ny * ua[:, 1]) + ha_w * np.abs(nx * va[:, 0] + ny * va[:, 1])
        rb = hb_l * np.abs(nx * ub[:, 0] + ny * ub[:, 1]) + hb_w * np.abs(nx * vb[:, 0] + ny * vb[:, 1])
        h = ra + rb
        d = nx * dx + ny * dy
        w = nx * vx + ny * vy
        gap = np.maximum(gap, np.abs(d) - h)

        static = np.abs(w) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h - d) / w
            t2 = (h - d) / w
        inside = np.abs(d) <= h
        lo = np.where(static, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        hi = np.where(static, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        t_enter = np.maximum(t_enter, lo)
        t_exit = np.minimum(t_exit, hi)

    hit = (t_enter <= t_exit) & (t_exit >= 0)
    return np.where(hit, np.maximum(t_enter, 0.0), np.inf), gap


def _ray_box(px, py, wx, wy, ex, ey):
    """First time a point p + w*t is inside the box |x|<=ex, |y|<=ey (0 if already inside, inf if never)."""
    t_in = np.full(px.shape, -np.inf)
    t_out = np.full(px.shape, np.inf)
    for p, w, e in ((px, wx, ex), (py, wy, ey)):
        static = np.abs(w) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-e - p) / w
            t2 = (e - p) / w
        inside = np.abs(p) <= e
        lo = np.where(static, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        hi = np.where(static, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        t_in = np.maximum(t_in, lo)
        t_out = np.minimum(t_out, hi)
    hit = (t_in <= t_out) & (t_out >= 0)
    return np.where(hit, np.maximum(t_in, 0.0), np.inf)


def _ray_circle(px, py, wx, wy, cx, cy, r):
    """First time a point p + w*t is inside the circle (centre c, radius r)."""
    qx, qy = px - cx, py - cy
    a = wx ** 2 + wy ** 2
    b = 2 * (qx * wx + qy * wy)
    c = qx ** 2 + qy ** 2 - r ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - np.sqrt(disc)) / (2 * a)
    ok = (a > 1e-12) & (b < 0) & (disc >= 0)
    return np.where(c <= 0, 0.0, np.where(ok, np.maximum(t, 0.0), np.inf))


def ttc_rect_circle(p, r=PED_RADIUS):
    """Vehicle (a, rectangle) vs pedestrian (b, circle). Exact. Returns ttc (inf if none), gap."""
    th = p["theta_a"].to_numpy()
    c, s = np.cos(th), np.sin(th)
    hl, hw = p["length_a"].to_numpy() / 2, p["width_a"].to_numpy() / 2
    dx, dy = p["x_b"].to_numpy() - p["x_a"].to_numpy(), p["y_b"].to_numpy() - p["y_a"].to_numpy()
    rvx = p["vx_b"].to_numpy() - p["vx_a"].to_numpy()
    rvy = p["vy_b"].to_numpy() - p["vy_a"].to_numpy()
    # into the vehicle's frame (rotate by -theta); vehicle sits at the origin
    px, py = c * dx + s * dy, -s * dx + c * dy
    wx, wy = c * rvx + s * rvy, -s * rvx + c * rvy

    # exact distance from the pedestrian's centre to the rectangle, minus the radius
    qx, qy = np.abs(px) - hl, np.abs(py) - hw
    sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
    gap = sd - r

    # rectangle grown by r = 2 boxes + 4 corner circles
    t = np.minimum(_ray_box(px, py, wx, wy, hl + r, hw), _ray_box(px, py, wx, wy, hl, hw + r))
    for sx in (-1, 1):
        for sy in (-1, 1):
            t = np.minimum(t, _ray_circle(px, py, wx, wy, sx * hl, sy * hw, r))
    return t, gap


def ttc_circle_circle(p, r=PED_RADIUS):
    t = _ray_circle(p["x_b"].to_numpy() - p["x_a"].to_numpy(), p["y_b"].to_numpy() - p["y_a"].to_numpy(),
                    p["vx_b"].to_numpy() - p["vx_a"].to_numpy(), p["vy_b"].to_numpy() - p["vy_a"].to_numpy(),
                    0.0, 0.0, 2 * r)
    gap = np.hypot(p["x_b"] - p["x_a"], p["y_b"] - p["y_a"]).to_numpy() - 2 * r
    return t, gap


def add_ttc(p, fn):
    t, g = fn(p)
    p = p.copy()
    p["ttc"], p["gap"] = t, g
    p["overlap"] = g <= 0
    dx, dy = p["x_b"] - p["x_a"], p["y_b"] - p["y_a"]
    dist = np.hypot(dx, dy).clip(lower=1e-9)
    p["closing_speed"] = -((dx * (p["vx_b"] - p["vx_a"]) + dy * (p["vy_b"] - p["vy_a"])) / dist)
    if MIN_CLOSING_SPEED > 0:   # optional noise filter on very slow closing
        p.loc[(p["closing_speed"] < MIN_CLOSING_SPEED) & ~p["overlap"], "ttc"] = np.inf
    return p


# ----------------------------------------------------------------------------- summaries
def per_agent_summary(frames, all_agents):
    """Long format (every agent appears once as 'self', once as 'other') -> per agent stats."""
    a = frames.rename(columns=lambda c: c.replace("_a", "_self").replace("_b", "_other"))
    b = frames.rename(columns=lambda c: c.replace("_b", "_self").replace("_a", "_other"))
    long = pd.concat([a, b], ignore_index=True)
    long = long[long["ttc"] <= TTC_HORIZON]

    idx = long.groupby("id_self")["ttc"].idxmin()
    low = long.loc[idx, ["id_self", "ttc", "frame_id", "timestamp_ms_self", "id_other",
                         "agent_type_other", "gap", "closing_speed", "overlap"]]
    low.columns = ["id", "min_ttc_s", "frame_at_min", "time_ms_at_min", "partner_id",
                   "partner_type", "gap_m_at_min", "closing_speed_ms_at_min", "overlap_at_min"]

    out = all_agents.merge(low, on="id", how="left")
    for th in THRESHOLDS:
        n = long[long["ttc"] < th].groupby("id_self").size().rename(f"time_below_{th}s")
        out = out.merge((n * FRAME_DT).reset_index().rename(columns={"id_self": "id"}),
                        on="id", how="left")
        out[f"time_below_{th}s"] = out[f"time_below_{th}s"].fillna(0.0)
    p5 = long.groupby("id_self")["ttc"].quantile(0.05).rename("ttc_p05_s").reset_index()
    return out.merge(p5.rename(columns={"id_self": "id"}), on="id", how="left")


def main():
    veh, ped = load()
    agents = pd.concat([veh, ped])
    all_agents = (agents.groupby("id")
                  .agg(kind=("kind", "first"), agent_type=("agent_type", "first"),
                       first_frame=("frame_id", "min"), last_frame=("frame_id", "max"),
                       n_frames=("frame_id", "size"))
                  .reset_index())

    # ---- optional: drop long-stationary (parked) vehicles
    all_agents["n_frames_excluded_stationary"] = 0
    if EXCLUDE_LONG_STATIONARY:
        flag = flag_long_stationary(veh)
        ex = veh[flag]
        if len(ex):
            summ = (ex.groupby("id").agg(agent_type=("agent_type", "first"),
                                         excluded_frames=("frame_id", "size"),
                                         first_frame=("frame_id", "min"), last_frame=("frame_id", "max"))
                    .assign(excluded_s=lambda d: (d["excluded_frames"] * FRAME_DT).round(1))
                    .reset_index())
            summ.to_csv(f"{OUT_DIR}/excluded_stationary.csv", index=False)
            all_agents = all_agents.merge(
                summ[["id", "excluded_frames"]].rename(columns={"excluded_frames": "n_frames_excluded_stationary_new"}),
                on="id", how="left")
            all_agents["n_frames_excluded_stationary"] = all_agents.pop("n_frames_excluded_stationary_new").fillna(0).astype(int)
        print(f"Excluded {flag.sum():,} stationary vehicle-frames from {ex['id'].nunique()} vehicles "
              f"(stationary > {STATIONARY_MAX_S:.0f} s at < {STOP_SPEED} m/s)")
        veh = veh[~flag]

    vv, vp, pp = build_pairs(veh, ped)
    print(f"{len(vv) + len(vp) + (0 if pp is None else len(pp)):,} pair-frames within {MAX_DIST} m")

    parts = [add_ttc(vv, ttc_rect_rect), add_ttc(vp, ttc_rect_circle)]
    if pp is not None:
        parts.append(add_ttc(pp, ttc_circle_circle))
    pairs = pd.concat(parts, ignore_index=True)

    frames = pairs[pairs["ttc"] <= TTC_HORIZON].copy()
    print(f"{len(frames):,} pair-frames with TTC <= {TTC_HORIZON}s")

    keep = ["frame_id", "timestamp_ms_a", "id_a", "id_b", "agent_type_a", "agent_type_b",
            "ttc", "gap", "closing_speed", "overlap", "x_a", "y_a", "x_b", "y_b"]
    frames[frames["ttc"] <= TTC_FRAME_EXPORT][keep].to_csv(f"{OUT_DIR}/ttc_frames.csv", index=False)

    idx = frames.groupby(["id_a", "id_b"])["ttc"].idxmin()
    pr = frames.loc[idx, ["id_a", "id_b", "agent_type_a", "agent_type_b", "ttc", "frame_id",
                          "gap", "closing_speed", "overlap"]].sort_values("ttc")
    pr.columns = ["id_a", "id_b", "type_a", "type_b", "min_ttc_s", "frame_at_min",
                  "gap_m_at_min", "closing_speed_ms_at_min", "overlap_at_min"]
    pr.to_csv(f"{OUT_DIR}/ttc_per_pair.csv", index=False)

    agent_table = per_agent_summary(frames, all_agents)
    agent_table.insert(1, "track_num", agent_table["id"].str.extract(r"(\d+)")[0].astype(int))
    agent_table["_is_ped"] = agent_table["id"].str.startswith("P")
    agent_table = agent_table.sort_values(["_is_ped", "track_num"]).drop(columns="_is_ped")
    agent_table.to_csv(f"{OUT_DIR}/ttc_per_agent.csv", index=False)
    print(agent_table.head(12).to_string())
    return agent_table, pr, frames


if __name__ == "__main__":
    main()
