import json

# File paths for the two JSON files
file2 = "data_coolinarika/data.json"
file1 = "data_recepti/recepti.json"
output_file = "merged.json"


# Load data from the first file
with open(file1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

# Load data from the second file
with open(file2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# Merge the "dishes" lists
merged_data = {"dishes": data1["dishes"] + data2["dishes"]}

# Save the merged data to a new file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"Data merged successfully into {output_file}")
