import pandas as pd
d = r"SinD\Data\Tianjin\8_2_1"
a = pd.read_csv(r"SinD\my_Veh_tracks_meta.csv")
b = pd.read_csv(d + r"\Veh_tracks_meta.csv")
m = a.merge(b, on="trackId", suffixes=("_me", "_ref"))
print(pd.crosstab(m.CrossType_me, m.CrossType_ref))
print(pd.crosstab(m.Signal_Violation_Behavior_me, m.Signal_Violation_Behavior_ref))
print((m.initialFrame_me == m.initialFrame_ref).mean(), (m.finalFrame_me == m.finalFrame_ref).mean())
print((m.length_me - m.length_ref).abs().describe())   # tells you mean vs median

v = pd.read_csv(d + r"\Veh_smoothed_tracks.csv", usecols=["frame_id", "timestamp_ms"])
print((v.timestamp_ms - v.frame_id * 3 / 29.97 * 1000).abs().max())


d = r"SinD\Data\Tianjin\8_2_1"
v = pd.read_csv(d + r"\Veh_smoothed_tracks.csv", usecols=["frame_id", "timestamp_ms"])
n = v.frame_id.max() - v.frame_id.min() + 1
print(n, "frames,", round((v.timestamp_ms.max() - v.timestamp_ms.min()) / 60000, 1), "min")
print(v.frame_id.nunique(), "frames that contain at least one vehicle")