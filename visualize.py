import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Elérési utak automatikus kezelése
my_path = (
    r"SinD\my_Veh_tracks_meta.csv"
    if os.path.exists(r"SinD\my_Veh_tracks_meta.csv")
    else "my_Veh_tracks_meta.csv"
)
ref_path = (
    r"SinD\Data\Tianjin\8_2_1\Veh_tracks_meta.csv"
    if os.path.exists(r"SinD\Data\Tianjin\8_2_1\Veh_tracks_meta.csv")
    else r"Data\Tianjin\8_2_1\Veh_tracks_meta.csv"
)

# 2. Beolvasás és összekapcsolás trackId alapján
df_me = pd.read_csv(my_path)
df_ref = pd.read_csv(ref_path)
m = df_me.merge(df_ref, on="trackId", suffixes=("_me", "_ref"))

modal_col = "class_ref"
signal_col = "Signal_Violation_Behavior_ref"

# 3. Kereszttábla készítése (darabszámok és arányok)
ct_counts = pd.crosstab(m[modal_col], m[signal_col])
ct_percent = pd.crosstab(m[modal_col], m[signal_col], normalize="index") * 100

print("--- Counts by Vehicle Class ---")
print(ct_counts)
print("\n--- Percentage Distribution (%) ---")
print(ct_percent.round(1))

# 4. Diagramok rajzolása angol feliratokkal
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

color_dict = {
    "No violation of traffic lights": "#2ecc71",  # Zöld: No violation
    "red-light running": "#e74c3c",  # Piros: Red-light running
    "yellow-light running": "#f39c12",  # Narancs: Yellow-light running
}
colors = [color_dict.get(col, "#95a5a6") for col in ct_percent.columns]

# BAL OLDALI ÁBRA: 100% Stacked Bar Chart (Arányok)
ct_percent.plot(
    kind="bar",
    stacked=True,
    color=colors,
    ax=axes[0],
    edgecolor="black",
    width=0.6,
)
axes[0].set_title(
    "Traffic Signal Compliance by Modality (100%)",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_xlabel("Vehicle Modality (Class)", fontsize=10)
axes[0].set_ylabel("Proportion (%)", fontsize=10)
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Behavior", bbox_to_anchor=(1.02, 1), loc="upper left")

# Százalékos címkék ráírása a sávokra
for p in axes[0].patches:
    height = p.get_height()
    if height > 4:
        axes[0].text(
            p.get_x() + p.get_width() / 2,
            p.get_y() + height / 2,
            f"{height:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

# JOBB OLDALI ÁBRA: Csoportosított oszlopdiagram (Darabszámok)
ct_counts.plot(
    kind="bar",
    color=colors,
    ax=axes[1],
    edgecolor="black",
    width=0.7,
)
axes[1].set_title(
    "Absolute Frequency of Violations by Modality",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel("Vehicle Modality (Class)", fontsize=10)
axes[1].set_ylabel("Vehicle Count", fontsize=10)
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Behavior", bbox_to_anchor=(1.02, 1), loc="upper left")

# Darabszám ráírása az oszlopok tetejére
for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

plt.tight_layout()
plt.savefig("modal_violation_chart_en.png", dpi=300)
print("\nChart successfully saved as: modal_violation_chart_en.png")
plt.show()