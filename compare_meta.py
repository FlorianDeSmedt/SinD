#!/usr/bin/env python3
"""
Compare a rebuilt Veh_tracks_meta.csv with the original and list every trackId that was labelled differently.

  python compare_meta.py --mine my_Veh_tracks_meta.csv --ref Data/Tianjin/8_2_1/Veh_tracks_meta.csv
  python compare_meta.py --mine my.csv --ref ref.csv --out diffs.csv --replay-cmd "python replay_tracks.py ... "

Checks CrossType and Signal_Violation_Behavior (and, as a sanity check, class / initialFrame / finalFrame).
Prints a summary, a confusion table per label, the list of differing ids, and writes them all to a CSV.

python compare_meta.py --mine my_Veh_tracks_meta.csv --ref Data/Tianjin/8_2_1/Veh_tracks_meta.csv --out diffs.csv
"""
import argparse
import pandas as pd

LABELS = ["CrossType", "Signal_Violation_Behavior"]
SANITY = ["class", "initialFrame", "finalFrame"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True, help="your rebuilt meta file")
    ap.add_argument("--ref", required=True, help="the original meta file")
    ap.add_argument("--out", default="meta_differences.csv")
    ap.add_argument("--replay-cmd", help="optional replay command prefix; prints '<prefix> --focus <id>' for each differing track")
    ap.add_argument("--max-list", type=int, default=60, help="how many differing ids to print (all go to the CSV)")
    a = ap.parse_args()

    mine = pd.read_csv(a.mine)
    ref = pd.read_csv(a.ref)
    for name, df in (("mine", mine), ("ref", ref)):
        if df.trackId.duplicated().any():
            print(f"warning: duplicate trackIds in {name}")

    only_mine = sorted(set(mine.trackId) - set(ref.trackId))
    only_ref = sorted(set(ref.trackId) - set(mine.trackId))
    m = ref.merge(mine, on="trackId", suffixes=("_ref", "_mine"))
    print(f"tracks: ref {len(ref)}, mine {len(mine)}, in both {len(m)}")
    if only_ref:  print(f"only in ref  ({len(only_ref)}): {only_ref[:20]}{' ...' if len(only_ref) > 20 else ''}")
    if only_mine: print(f"only in mine ({len(only_mine)}): {only_mine[:20]}{' ...' if len(only_mine) > 20 else ''}")

    # sanity columns first: if these differ the files are not comparable one-to-one
    for c in SANITY:
        if f"{c}_ref" in m and f"{c}_mine" in m:
            bad = (m[f"{c}_ref"] != m[f"{c}_mine"]).sum()
            if bad:
                print(f"note: {bad} tracks differ in '{c}' (not a labelling difference - check the inputs)")

    diff = pd.Series(False, index=m.index)
    for c in LABELS:
        m[f"{c}_differs"] = m[f"{c}_ref"].astype(str) != m[f"{c}_mine"].astype(str)
        diff |= m[f"{c}_differs"]
        agree = 100 * (1 - m[f"{c}_differs"].mean())
        print(f"\n{c}: {int(m[f'{c}_differs'].sum())} differ  ({agree:.1f}% agreement)")
        if m[f"{c}_differs"].any():
            print(pd.crosstab(m[f"{c}_mine"], m[f"{c}_ref"], rownames=["mine"], colnames=["ref"]).to_string())

    d = m[diff].copy()
    def kind(r):
        k = [c for c in LABELS if r[f"{c}_differs"]]
        return "+".join(k)
    d["differs_in"] = d.apply(kind, axis=1)
    cols = ["trackId", "class_ref", "initialFrame_ref", "finalFrame_ref", "differs_in"]
    for c in LABELS:
        cols += [f"{c}_ref", f"{c}_mine"]
    d = d[cols].sort_values(["differs_in", "trackId"])
    d.to_csv(a.out, index=False)

    print(f"\n{len(d)} tracks labelled differently -> {a.out}")
    if len(d):
        short = d.rename(columns={"class_ref": "class", "initialFrame_ref": "from", "finalFrame_ref": "to",
                                  "CrossType_ref": "Cross_ref", "CrossType_mine": "Cross_mine",
                                  "Signal_Violation_Behavior_ref": "Viol_ref", "Signal_Violation_Behavior_mine": "Viol_mine"})
        print(short.head(a.max_list).to_string(index=False))
        if len(d) > a.max_list:
            print(f"... {len(d) - a.max_list} more in the CSV")
        if a.replay_cmd:
            print("\nreplay each (first 10):")
            for tid in d.trackId.head(10):
                print(f"{a.replay_cmd} --focus {tid} --out track{tid}.mp4")


if __name__ == "__main__":
    main()
