import pandas as pd

# Load the vehicle trajectory data
df = pd.read_csv("Data/Tianjin/8_2_1/Veh_smoothed_tracks.csv")

# Count the number of unique vehicles per category
vehicle_counts = df.groupby("agent_type")["track_id"].nunique()

print(vehicle_counts)
