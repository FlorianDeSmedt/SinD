"""
Average time spent below 1 s TTC, per vehicle type, split by traffic-light violation.

Needs
  * ttc_per_agent.csv  (from compute_ttc.py)
  * per_voertuig.csv   (your per-vehicle table with the violation labels)

Outputs (OUT_DIR)
  summary_by_type_and_violation.csv   n, mean, SD, SEM, % of vehicles with any time below the threshold
  fig1_mean_time_below_1s.png         mean seconds below 1 s TTC per vehicle (bars = mean, whiskers = +/- 1 SEM)
  fig2_share_with_any_time_below_1s.png   % of vehicles that were below 1 s TTC at least once
  fig3_individual_vehicles.png        every single vehicle (jittered points) + group mean

Install matplotlib once if needed:  pip install matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write PNG files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent                       # the folder containing this script (SinD)
VIOLATION_FILE = BASE / "analysis_outputs" / "lanelet_audit" /"tianjin_core_v4" / "per_voertuig.csv"                   # <- change if the file lives elsewhere
TTC_FILE = BASE / "outputs_ttc" / "ttc_per_agent.csv"
OUT_DIR = BASE / "outputs_ttc" / "violation_analysis"

METRIC = "time_below_1.0s"      # column of ttc_per_agent.csv: seconds spent below 1 s TTC
                                # (alternatives: time_below_1.5s, time_below_3.0s)
THRESHOLD_LABEL = "1 s"         # only used in titles / labels

# How to split the vehicles:
#   "annotation"   -> meta_Signal_Violation_Behavior: no violation / red-light running / yellow-light running
#   "detected_red" -> signal_status == suspected_red_light_violation  vs  everything else
GROUPING = "annotation"

DROP_LONG_STATIONARY = True     # drop vehicles that were (partly) excluded as parked by compute_ttc.py
MIN_N_WARN = 10                 # groups with fewer vehicles are drawn faded/hatched (too few to trust)

GROUPS = {
    "annotation": {
        "No violation of traffic lights": ("No violation", "#7f8c9b"),
        "red-light running": ("Red-light running", "#d62728"),
        "yellow-light running": ("Yellow-light running", "#e0a800"),
    },
    "detected_red": {
        False: ("No suspected red-light violation", "#7f8c9b"),
        True: ("Suspected red-light violation", "#d62728"),
    },
}[GROUPING]


# ----------------------------------------------------------------------------- data
def load():
    viol = pd.read_csv(VIOLATION_FILE, encoding="utf-8-sig")      # utf-8-sig removes the hidden BOM in 'track_id'
    ttc = pd.read_csv(TTC_FILE)

    ttc = ttc[ttc["kind"] == "vehicle"].copy()
    ttc["track_id"] = ttc["id"].str.extract(r"(\d+)")[0].astype(int)

    df = viol.merge(ttc[["track_id", METRIC, "n_frames_excluded_stationary"]], on="track_id", how="left")
    missing = df[METRIC].isna().sum()
    if missing:
        print(f"WARNING: {missing} vehicles in the violation file have no TTC row and are dropped")
        df = df.dropna(subset=[METRIC])

    if DROP_LONG_STATIONARY:
        n_drop = (df["n_frames_excluded_stationary"] > 0).sum()
        df = df[df["n_frames_excluded_stationary"] == 0]
        print(f"Dropped {n_drop} long-stationary (parked) vehicles")

    if GROUPING == "annotation":
        df["violation_key"] = df["meta_Signal_Violation_Behavior"]
    else:
        df["violation_key"] = df["signal_status"].eq("suspected_red_light_violation")
    df["violation"] = df["violation_key"].map(lambda k: GROUPS[k][0] if k in GROUPS else np.nan)
    return df.dropna(subset=["violation"])


def summarise(df):
    both = pd.concat([df, df.assign(agent_type="all vehicles")], ignore_index=True)
    g = both.groupby(["agent_type", "violation"])[METRIC]
    t = g.agg(n="size", mean_s="mean", sd_s="std", n_any=lambda s: int((s > 0).sum())).reset_index()
    t["sem_s"] = t["sd_s"] / np.sqrt(t["n"])
    t["pct_any"] = 100 * t["n_any"] / t["n"]
    p = t["n_any"] / t["n"]
    t["pct_any_se"] = 100 * np.sqrt(p * (1 - p) / t["n"])
    return t


# ----------------------------------------------------------------------------- plots
def type_order(df):
    return list(df["agent_type"].value_counts().index) + ["all vehicles"]


def grouped_bar(tab, types, value, err, ylabel, title, path):
    labels = [v[0] for v in GROUPS.values()]
    colours = {v[0]: v[1] for v in GROUPS.values()}
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    top = 0
    for i, lab in enumerate(labels):
        sub = tab[tab["violation"] == lab].set_index("agent_type").reindex(types)
        x = np.arange(len(types)) + (i - (len(labels) - 1) / 2) * width
        vals = sub[value].to_numpy(dtype=float)
        errs = sub[err].to_numpy(dtype=float)
        small = (sub["n"].fillna(0) < MIN_N_WARN).to_numpy()
        for xi, v, e, sm, n in zip(x, vals, errs, small, sub["n"].fillna(0)):
            if n == 0 or np.isnan(v):
                continue
            ax.bar(xi, v, width * 0.92, color=colours[lab], alpha=0.35 if sm else 0.95,
                   hatch="//" if sm else None, edgecolor="white" if not sm else colours[lab], linewidth=0.8)
            e = 0 if np.isnan(e) else e
            ax.errorbar(xi, v, yerr=[[min(e, v)], [e]], color="black", capsize=3, lw=1)
            ax.text(xi, v + e + 0.01 * max(1e-9, np.nanmax(vals)), f"n={int(n)}", ha="center", va="bottom",
                    fontsize=7.5, color="#333")
            top = max(top, v + e)
        ax.bar(np.nan, np.nan, color=colours[lab], label=lab)      # legend entry
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types)
    ax.axvline(len(types) - 1.5, color="grey", lw=0.8, ls=":")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, top * 1.18 if top > 0 else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.01, 0.005, f"Hatched/faded bars: fewer than {MIN_N_WARN} vehicles in the group. "
                          "Whiskers: +/- 1 standard error.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def strip_plot(df, types, path):
    labels = [v[0] for v in GROUPS.values()]
    colours = {v[0]: v[1] for v in GROUPS.values()}
    types = [t for t in types if t != "all vehicles"]
    width = 0.8 / len(labels)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, lab in enumerate(labels):
        for j, t in enumerate(types):
            vals = df[(df["agent_type"] == t) & (df["violation"] == lab)][METRIC].to_numpy()
            if len(vals) == 0:
                continue
            x0 = j + (i - (len(labels) - 1) / 2) * width
            ax.scatter(x0 + rng.uniform(-width * 0.35, width * 0.35, len(vals)), vals, s=14,
                       color=colours[lab], alpha=0.45, linewidths=0)
            ax.hlines(vals.mean(), x0 - width * 0.42, x0 + width * 0.42, color="black", lw=2)
        ax.scatter([], [], color=colours[lab], label=lab)
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types)
    ax.set_ylabel(f"Seconds below {THRESHOLD_LABEL} TTC (per vehicle)")
    ax.set_title(f"Every vehicle: time below {THRESHOLD_LABEL} TTC (black line = group mean)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"{len(df)} vehicles analysed")
    print(df.groupby(["agent_type", "violation"]).size().unstack(fill_value=0).to_string(), "\n")

    tab = summarise(df)
    types = type_order(df)
    tab["agent_type"] = pd.Categorical(tab["agent_type"], types, ordered=True)
    tab = tab.sort_values(["agent_type", "violation"])
    tab.round(4).to_csv(OUT_DIR / "summary_by_type_and_violation.csv", index=False)
    print(tab.round(3).to_string(index=False))

    grouped_bar(tab, types, "mean_s", "sem_s",
                f"Mean seconds below {THRESHOLD_LABEL} TTC per vehicle",
                f"Average time spent below {THRESHOLD_LABEL} TTC, by vehicle type and light violation",
                OUT_DIR / "fig1_mean_time_below_1s.png")
    grouped_bar(tab, types, "pct_any", "pct_any_se",
                f"% of vehicles with any time below {THRESHOLD_LABEL} TTC",
                f"Share of vehicles that were below {THRESHOLD_LABEL} TTC at least once",
                OUT_DIR / "fig2_share_with_any_time_below_1s.png")
    strip_plot(df, types, OUT_DIR / "fig3_individual_vehicles.png")
    print(f"\nSaved table and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
