import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

# --- CONFIGURATION ---
TRACKS_PATH = os.path.join("Data", "Tianjin", "8_2_1", "Veh_smoothed_tracks.csv")
META_PATH = os.path.join("Data", "Tianjin", "8_2_1", "Veh_tracks_meta.csv")
OUTPUT_DIR = "outputs_pet"

MIN_PET_SEC = -0.5          # Filter out stationary/waiting overlaps (no actual crashes in dataset)
MAX_PET_SEC = 10.0          # Maximum time window (seconds) to consider a crossing interaction
MIN_HEADING_DIFF_DEG = 25.0 # Exclude car-following in the same lane
MIN_SPEED_MPS = 0.5         # Minimum average speed (m/s) in conflict zone to exclude parked/waiting vehicles


def create_vehicle_polygon(x: float, y: float, length: float, width: float, yaw_rad: float) -> Polygon:
    hl = length / 2.0
    hw = width / 2.0
    box = Polygon([(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)])
    rotated = rotate(box, yaw_rad, origin=(0, 0), use_radians=True)
    return translate(rotated, xoff=x, yoff=y)


def angle_diff_deg(rad1: float, rad2: float) -> float:
    diff = abs((rad1 - rad2 + math.pi) % (2 * math.pi) - math.pi)
    return math.degrees(diff)


def load_data():
    print("Loading smoothed tracks and metadata...")
    tracks = pd.read_csv(TRACKS_PATH)
    meta = pd.read_csv(META_PATH)

    tracks["t_sec"] = tracks["timestamp_ms"] / 1000.0
    tracks["speed"] = np.hypot(tracks["vx"], tracks["vy"])

    meta_subset = meta[["trackId", "CrossType", "Signal_Violation_Behavior"]].rename(
        columns={"trackId": "track_id"}
    )
    tracks = tracks.merge(meta_subset, on="track_id", how="left")
    tracks["Signal_Violation_Behavior"] = tracks["Signal_Violation_Behavior"].fillna("Unknown")
    return tracks


def precompute_trajectories(tracks: pd.DataFrame) -> dict:
    print("Precomputing 2D oriented vehicle footprints...")
    veh_data = {}

    for tid, df in tracks.groupby("track_id"):
        df = df.sort_values("t_sec").reset_index(drop=True)
        times = df["t_sec"].to_numpy()
        xs = df["x"].to_numpy()
        ys = df["y"].to_numpy()
        yaws = df["yaw_rad"].to_numpy()
        speeds = df["speed"].to_numpy()
        lengths = df["length"].to_numpy()
        widths = df["width"].to_numpy()

        polys = [
            create_vehicle_polygon(x, y, l, w, yaw)
            for x, y, l, w, yaw in zip(xs, ys, lengths, widths, yaws)
        ]
        swept_area = unary_union(polys)

        veh_data[tid] = {
            "track_id": tid,
            "agent_type": df["agent_type"].iloc[0],
            "violation": df["Signal_Violation_Behavior"].iloc[0],
            "cross_type": df["CrossType"].iloc[0],
            "t_start": times[0],
            "t_end": times[-1],
            "times": times,
            "yaws": yaws,
            "speeds": speeds,
            "polys": polys,
            "swept_area": swept_area,
            "bounds": swept_area.bounds,
        }
    return veh_data


def bboxes_overlap(b1, b2) -> bool:
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def compute_pair_pet(v1: dict, v2: dict):
    if v1["t_end"] + MAX_PET_SEC < v2["t_start"] or v2["t_end"] + MAX_PET_SEC < v1["t_start"]:
        return None

    if not bboxes_overlap(v1["bounds"], v2["bounds"]):
        return None

    conflict_zone = v1["swept_area"].intersection(v2["swept_area"])
    if conflict_zone.is_empty or conflict_zone.area < 0.05:
        return None

    idx1 = [i for i, p in enumerate(v1["polys"]) if p.intersects(conflict_zone)]
    idx2 = [i for i, p in enumerate(v2["polys"]) if p.intersects(conflict_zone)]

    if not idx1 or not idx2:
        return None

    # Exclude stationary/waiting vehicles inside the conflict zone
    if np.mean([v1["speeds"][i] for i in idx1]) < MIN_SPEED_MPS:
        return None
    if np.mean([v2["speeds"][i] for i in idx2]) < MIN_SPEED_MPS:
        return None

    mean_yaw1 = np.mean([v1["yaws"][i] for i in idx1])
    mean_yaw2 = np.mean([v2["yaws"][i] for i in idx2])
    heading_diff = angle_diff_deg(mean_yaw1, mean_yaw2)
    if heading_diff < MIN_HEADING_DIFF_DEG:
        return None

    t1_enter, t1_exit = v1["times"][idx1[0]], v1["times"][idx1[-1]]
    t2_enter, t2_exit = v2["times"][idx2[0]], v2["times"][idx2[-1]]

    if t1_enter <= t2_enter:
        first, second = v1, v2
        t_first_exit, t_second_enter = t1_exit, t2_enter
    else:
        first, second = v2, v1
        t_first_exit, t_second_enter = t2_exit, t1_enter

    pet = t_second_enter - t_first_exit

    # Keep only valid crossing interactions within [MIN_PET_SEC, MAX_PET_SEC]
    if pet < MIN_PET_SEC or pet > MAX_PET_SEC:
        return None

    simultaneous_in_intersection = max(v1["t_start"], v2["t_start"]) <= min(v1["t_end"], v2["t_end"])

    return {
        "first_track_id": first["track_id"],
        "first_agent_type": first["agent_type"],
        "first_violation": first["violation"],
        "second_track_id": second["track_id"],
        "second_agent_type": second["agent_type"],
        "second_violation": second["violation"],
        "pet_sec": round(pet, 3),
        "t_first_exit_sec": round(t_first_exit, 3),
        "t_second_enter_sec": round(t_second_enter, 3),
        "heading_diff_deg": round(heading_diff, 1),
        "simultaneous_presence": simultaneous_in_intersection,
    }


def generate_plots(df_pet: pd.DataFrame):
    print("Generating PET charts...")
    sns.set_theme(style="whitegrid", context="talk")

    # 1. Plot: PET distribution by Encroaching Vehicle Type
    order_types = df_pet["second_agent_type"].value_counts().index
    plt.figure(figsize=(11, 6))
    sns.boxplot(
        data=df_pet,
        x="second_agent_type",
        y="pet_sec",
        hue="second_agent_type",
        order=order_types,
        palette="Set2",
        showfliers=False,
        legend=False,
    )
    sns.stripplot(
        data=df_pet,
        x="second_agent_type",
        y="pet_sec",
        order=order_types,
        color="black",
        alpha=0.25,
        size=4,
        jitter=True,
    )
    plt.axhline(1.5, color="red", linestyle="--", linewidth=1.5, label="Critical PET threshold (1.5 s)")
    plt.title("Post Encroachment Time (PET) by Encroaching Vehicle Type (Tianjin 8_2_1)", pad=35)
    plt.xlabel("Encroaching Vehicle Type")
    plt.ylabel("PET (seconds)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pet_by_vehicle_type.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Plot: PET comparison — Red-Light Running vs. Compliant Crossing
    top_modes = ["car", "motorcycle", "bicycle", "tricycle"]
    df_sub = df_pet[
        (df_pet["second_agent_type"].isin(top_modes)) &
        (df_pet["second_violation"].isin(["No violation of traffic lights", "red-light running"]))
    ].copy()

    df_sub["Violation Status"] = df_sub["second_violation"].replace({
        "No violation of traffic lights": "Compliant (Green)",
        "red-light running": "Red-Light Running"
    })

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_sub,
        x="second_agent_type",
        y="pet_sec",
        hue="Violation Status",
        order=top_modes,
        palette={"Compliant (Green)": "#4daf4a", "Red-Light Running": "#e41a1c"},
    )
    plt.axhline(1.5, color="black", linestyle="--", linewidth=1.5, label="Critical PET threshold (1.5 s)")
    plt.title("PET Comparison: Compliant vs. Red-Light Running by Vehicle Type", pad=38)
    plt.xlabel("Encroaching Vehicle Type")
    plt.ylabel("PET (seconds)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pet_red_light_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tracks = load_data()
    veh_data = precompute_trajectories(tracks)

    track_ids = sorted(veh_data.keys())
    n = len(track_ids)
    print(f"Evaluating pairwise conflicts across {n} vehicles...")

    results = []
    for i in range(n):
        v1 = veh_data[track_ids[i]]
        for j in range(i + 1, n):
            v2 = veh_data[track_ids[j]]
            res = compute_pair_pet(v1, v2)
            if res is not None:
                results.append(res)

    df_pet = pd.DataFrame(results)
    if df_pet.empty:
        print("No crossing conflicts found within the specified thresholds.")
        return

    df_pet.to_csv(os.path.join(OUTPUT_DIR, "pet_conflicts_8_2_1.csv"), index=False)

    summary_type = (
        df_pet.groupby("second_agent_type")["pet_sec"]
        .agg(count="count", mean_pet="mean", median_pet="median", min_pet="min", p15_pet=lambda x: np.percentile(x, 15))
        .round(3)
    )
    summary_type.to_csv(os.path.join(OUTPUT_DIR, "pet_by_agent_type.csv"))

    summary_violation = (
        df_pet.groupby(["second_violation", "second_agent_type"])["pet_sec"]
        .agg(count="count", mean_pet="mean", median_pet="median", min_pet="min")
        .round(3)
    )
    summary_violation.to_csv(os.path.join(OUTPUT_DIR, "pet_by_violation_and_type.csv"))

    generate_plots(df_pet)

    print(f"\nDone! Found {len(df_pet)} valid crossing conflict events.")
    print(f"Results and charts saved to '{OUTPUT_DIR}/'\n")
    print("--- PET Summary by Encroaching Vehicle Type (seconds) ---")
    print(summary_type)
    print("\n--- PET Summary by Signal Violation Behavior (seconds) ---")
    print(summary_violation)


if __name__ == "__main__":
    main()