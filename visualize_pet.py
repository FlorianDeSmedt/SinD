import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = os.path.join("outputs_pet", "pet_conflicts_8_2_1.csv")
OUTPUT_DIR = "outputs_pet"

def main():
    df = pd.read_csv(INPUT_CSV)
    sns.set_theme(style="whitegrid", context="talk")

    # 1. Plot: PET distribution by Encroaching Vehicle Type
    order_types = df["second_agent_type"].value_counts().index
    plt.figure(figsize=(11, 6))
    ax1 = sns.boxplot(
        data=df,
        x="second_agent_type",
        y="pet_sec",
        order=order_types,
        palette="Set2",
        showfliers=False,
    )
    sns.stripplot(
        data=df,
        x="second_agent_type",
        y="pet_sec",
        order=order_types,
        color="black",
        alpha=0.25,
        size=4,
        jitter=True,
    )
    plt.axhline(1.5, color="red", linestyle="--", linewidth=1.5, label="Critical PET threshold (1.5 s)")
    plt.title("Post Encroachment Time (PET) by Encroaching Vehicle Type (Tianjin 8_2_1)")
    plt.xlabel("Encroaching Vehicle Type")
    plt.ylabel("PET (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pet_by_vehicle_type.png"), dpi=300)
    plt.close()

    # 2. Plot: PET comparison — Red-Light Running vs. Compliant Crossing (top 4 modes)
    top_modes = ["car", "motorcycle", "bicycle", "tricycle"]
    df_sub = df[
        (df["second_agent_type"].isin(top_modes)) &
        (df["second_violation"].isin(["No violation of traffic lights", "red-light running"]))
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
    plt.title("PET Comparison: Compliant vs. Red-Light Running by Vehicle Type")
    plt.xlabel("Encroaching Vehicle Type")
    plt.ylabel("PET (seconds)")
    plt.legend(title="Signal Behavior")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pet_red_light_comparison.png"), dpi=300)
    plt.close()

    print("Charts saved to 'outputs_pet/pet_by_vehicle_type.png' and 'outputs_pet/pet_red_light_comparison.png'")

if __name__ == "__main__":
    main()