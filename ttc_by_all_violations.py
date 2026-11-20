"""
Time spent below a TTC threshold, per vehicle type, for EVERY violation type in per_voertuig.csv
(red-light running, yellow-light running, suspected red-light violation, wrong-way driving).

For each violation the script compares
    vehicles WITH that violation   versus   "clean" vehicles (no violation of any kind below)
per vehicle type, and writes a summary table + 3 figures, same layout as ttc_by_violation.py.
It also writes an overview comparing all violations side by side, including a check that corrects for
the vehicle-type mix (bicycles have much higher exposure than cars, so a group with many bicycles
looks worse even if the violation itself changes nothing).

Needs
  * outputs_ttc/ttc_per_agent.csv   (from compute_ttc.py)
  * per_voertuig.csv                (your per-vehicle table with the violation labels)

Outputs (OUT_DIR)
  overview_all_violations.csv / overview_all_violations.png
  <violation>_summary.csv, <violation>_fig1_mean.png, <violation>_fig2_share_any.png, <violation>_fig3_vehicles.png
  overlap_between_violations.csv     how many vehicles have several violations at once
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write PNG files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent                       # the folder containing this script (SinD)
VIOLATION_FILE = BASE / "analysis_outputs" / "lanelet_audit" / "tianjin_core_v4" / "per_voertuig.csv"
TTC_FILE = BASE / "outputs_ttc" / "ttc_per_agent.csv"
OUT_DIR = BASE / "outputs_ttc" / "violation_analysis_all"

METRIC = "time_below_1.0s"      # seconds below the threshold (alternatives: time_below_1.5s, time_below_3.0s)
THRESHOLD_LABEL = "1 s"         # only used in titles / labels
DROP_LONG_STATIONARY = True     # drop vehicles that compute_ttc.py excluded as parked
MIN_N_WARN = 10                 # groups with fewer vehicles are drawn faded/hatched
REF_LABEL = "No violation (reference)"
REF_COLOUR = "#7f8c9b"

# key: (label, colour, rule that is True for vehicles WITH this violation)
VIOLATIONS = {
    "red_light": ("Red-light running (annotation)", "#d62728",
                  lambda d: d["meta_Signal_Violation_Behavior"].eq("red-light running")),
    "yellow_light": ("Yellow-light running (annotation)", "#e0a800",
                     lambda d: d["meta_Signal_Violation_Behavior"].eq("yellow-light running")),
    "red_light_detected": ("Suspected red-light violation (own detection)", "#8c1c13",
                           lambda d: d["signal_status"].eq("suspected_red_light_violation")),
    "wrong_way": ("Wrong-way driving", "#6a3d9a",
                  lambda d: d["wrong_way_status"].eq("wrong_way_detected")),
}
ACTIVE = list(VIOLATIONS)       # remove keys here to analyse only some violations


# ----------------------------------------------------------------------------- data
def load():
    for f in (VIOLATION_FILE, TTC_FILE):
        if not f.exists():
            raise FileNotFoundError(f"Cannot find {f}\nCheck the path settings at the top of the script.")

    viol = pd.read_csv(VIOLATION_FILE, encoding="utf-8-sig")      # utf-8-sig removes the hidden BOM
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

    for k in ACTIVE:
        df[k] = VIOLATIONS[k][2](df).fillna(False).astype(bool)
    df["clean"] = ~df[ACTIVE].any(axis=1)
    return df.reset_index(drop=True)


def summarise(d):
    both = pd.concat([d, d.assign(agent_type="all vehicles")], ignore_index=True)
    g = both.groupby(["agent_type", "violation"])[METRIC]
    t = g.agg(n="size", mean_s="mean", sd_s="std", n_any=lambda s: int((s > 0).sum())).reset_index()
    t["sem_s"] = t["sd_s"] / np.sqrt(t["n"])
    t["pct_any"] = 100 * t["n_any"] / t["n"]
    p = t["n_any"] / t["n"]
    t["pct_any_se"] = 100 * np.sqrt(p * (1 - p) / t["n"])
    return t


# ----------------------------------------------------------------------------- plots per violation
def grouped_bar(tab, types, groups, value, err, ylabel, title, path):
    """groups: list of (label, colour)."""
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(types) + 2), 5.5))
    top = 0
    for i, (lab, col) in enumerate(groups):
        sub = tab[tab["violation"] == lab].set_index("agent_type").reindex(types)
        x = np.arange(len(types)) + (i - (len(groups) - 1) / 2) * width
        vals = sub[value].to_numpy(dtype=float)
        errs = sub[err].to_numpy(dtype=float)
        for xi, v, e, n in zip(x, vals, errs, sub["n"].fillna(0)):
            if n == 0 or np.isnan(v):
                continue
            sm = n < MIN_N_WARN
            ax.bar(xi, v, width * 0.92, color=col, alpha=0.35 if sm else 0.95,
                   hatch="//" if sm else None, edgecolor=col if sm else "white", linewidth=0.8)
            e = 0 if np.isnan(e) else e
            ax.errorbar(xi, v, yerr=[[min(e, v)], [e]], color="black", capsize=3, lw=1)
            ax.text(xi, v + e + 0.01 * max(1e-9, np.nanmax(vals)), f"n={int(n)}", ha="center",
                    va="bottom", fontsize=7.5, color="#333")
            top = max(top, v + e)
        ax.bar(np.nan, np.nan, color=col, label=lab)
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types)
    if "all vehicles" in types:
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


def strip_plot(d, types, groups, title, path):
    types = [t for t in types if t != "all vehicles"]
    width = 0.8 / len(groups)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(types) + 2), 5.5))
    for i, (lab, col) in enumerate(groups):
        for j, t in enumerate(types):
            vals = d[(d["agent_type"] == t) & (d["violation"] == lab)][METRIC].to_numpy()
            if len(vals) == 0:
                continue
            x0 = j + (i - (len(groups) - 1) / 2) * width
            ax.scatter(x0 + rng.uniform(-width * 0.35, width * 0.35, len(vals)), vals, s=14,
                       color=col, alpha=0.45, linewidths=0)
            ax.hlines(vals.mean(), x0 - width * 0.42, x0 + width * 0.42, color="black", lw=2)
        ax.scatter([], [], color=col, label=lab)
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types)
    ax.set_ylabel(f"Seconds below {THRESHOLD_LABEL} TTC (per vehicle)")
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyse_violation(df, key):
    label, colour, _ = VIOLATIONS[key]
    d = df[df[key] | df["clean"]].copy()
    d["violation"] = np.where(d[key], label, REF_LABEL)
    groups = [(REF_LABEL, REF_COLOUR), (label, colour)]

    # only show vehicle types that actually have vehicles with this violation
    have = set(d.loc[d[key], "agent_type"])
    order = [t for t in d["agent_type"].value_counts().index if t in have] + ["all vehicles"]

    tab = summarise(d)
    tab = tab[tab["agent_type"].isin(order)].copy()
    tab["agent_type"] = pd.Categorical(tab["agent_type"], order, ordered=True)
    tab = tab.sort_values(["agent_type", "violation"])
    tab.round(4).to_csv(OUT_DIR / f"{key}_summary.csv", index=False)

    grouped_bar(tab, order, groups, "mean_s", "sem_s", f"Mean seconds below {THRESHOLD_LABEL} TTC per vehicle",
                f"Time below {THRESHOLD_LABEL} TTC: {label} vs. no violation", OUT_DIR / f"{key}_fig1_mean.png")
    grouped_bar(tab, order, groups, "pct_any", "pct_any_se",
                f"% of vehicles with any time below {THRESHOLD_LABEL} TTC",
                f"Share of vehicles below {THRESHOLD_LABEL} TTC at least once: {label}",
                OUT_DIR / f"{key}_fig2_share_any.png")
    strip_plot(d, order, groups, f"Every vehicle: {label} (black line = group mean)",
               OUT_DIR / f"{key}_fig3_vehicles.png")
    return tab


# ----------------------------------------------------------------------------- overview
def overview(df):
    clean = df[df["clean"]]
    clean_mean_by_type = clean.groupby("agent_type")[METRIC].mean()

    rows = [("reference", REF_LABEL, REF_COLOUR, clean)]
    for k in ACTIVE:
        rows.append((k, VIOLATIONS[k][0], VIOLATIONS[k][1], df[df[k]]))
    rows.append(("any_violation", "Any of the violations", "#444444", df[~df["clean"]]))

    out = []
    for key, label, col, d in rows:
        x = d[METRIC]
        n = len(d)
        n_any = int((x > 0).sum())
        # expected mean if this group had the same time-below-threshold as clean vehicles of the SAME type
        exp = d["agent_type"].map(clean_mean_by_type).mean()
        p = n_any / n if n else np.nan
        out.append(dict(group=label, key=key, n=n, mean_s=x.mean(), sem_s=x.std() / np.sqrt(n) if n > 1 else np.nan,
                        n_any=n_any, pct_any=100 * p, pct_any_se=100 * np.sqrt(p * (1 - p) / n) if n else np.nan,
                        expected_mean_s_same_type_mix=exp,
                        ratio_observed_to_expected=x.mean() / exp if exp and exp > 0 else np.nan,
                        colour=col))
    return pd.DataFrame(out)


def overview_plot(ov, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    short = [textwrap.fill(g.replace(" (annotation)", "").replace(" (own detection)", " - own detection"), 15)
             for g in ov["group"]]
    x = np.arange(len(ov))
    for ax, val, err, ylab, ttl in (
            (axes[0], "mean_s", "sem_s", f"Mean seconds below {THRESHOLD_LABEL} TTC per vehicle",
             "Average time below threshold"),
            (axes[1], "pct_any", "pct_any_se", f"% of vehicles with any time below {THRESHOLD_LABEL} TTC",
             "Share of vehicles with any exposure")):
        for xi, (_, r) in zip(x, ov.iterrows()):
            sm = r["n"] < MIN_N_WARN
            ax.bar(xi, r[val], 0.7, color=r["colour"], alpha=0.35 if sm else 0.95,
                   hatch="//" if sm else None, edgecolor=r["colour"] if sm else "white")
            e = 0 if np.isnan(r[err]) else r[err]
            ax.errorbar(xi, r[val], yerr=[[min(e, r[val])], [e]], color="black", capsize=3, lw=1)
            ax.text(xi, r[val] + e, f"n={int(r['n'])}", ha="center", va="bottom", fontsize=8)
        if val == "mean_s":      # diamonds = what we would expect given the vehicle-type mix
            ax.scatter(x, ov["expected_mean_s_same_type_mix"], marker="D", s=40, facecolor="white",
                       edgecolor="black", zorder=5, label="Expected if same behaviour as clean\nvehicles of the same type")
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
        ax.axhline(ov.loc[0, val], color=REF_COLOUR, ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Time below {THRESHOLD_LABEL} TTC by violation (all vehicle types pooled)", fontsize=13)
    fig.text(0.01, 0.005, "Groups can overlap (a vehicle can run a red light AND drive the wrong way). "
                          f"Faded bars: fewer than {MIN_N_WARN} vehicles. Whiskers: +/- 1 SE.", fontsize=8,
             color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"{len(df)} vehicles analysed; {int(df['clean'].sum())} are 'clean' (no violation of any kind)\n")

    for k in ACTIVE:
        analyse_violation(df, k)
        print(f"  done: {VIOLATIONS[k][0]}  (n={int(df[k].sum())})")

    ov = overview(df)
    ov.drop(columns="colour").round(4).to_csv(OUT_DIR / "overview_all_violations.csv", index=False)
    overview_plot(ov, OUT_DIR / "overview_all_violations.png")
    print("\n" + ov.drop(columns=["colour", "key"]).round(3).to_string(index=False))

    ov_mat = pd.DataFrame({a: {b: int((df[a] & df[b]).sum()) for b in ACTIVE} for a in ACTIVE})
    ov_mat.to_csv(OUT_DIR / "overlap_between_violations.csv")
    print("\nVehicles with both violations (overlap):\n" + ov_mat.to_string())
    print(f"\nSaved everything to {OUT_DIR}")


if __name__ == "__main__":
    main()
