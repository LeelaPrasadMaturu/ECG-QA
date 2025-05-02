import os
import json

def merge_filtered_yes_no(source_folder, output_file):
    combined_data = []

    for filename in os.listdir(source_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(source_folder, filename)

            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {filename}")
                    continue

            filtered = [
                item for item in data
                if (
                    item.get("question_type")
                    and item["question_type"].lower() in ["single-verify"]
                )
            ]

            combined_data.extend(filtered)

    # Save to output file
    with open(output_file, 'w') as out_f:
        json.dump(combined_data, out_f, indent=4)

# Example usage:
merge_filtered_yes_no("./processed/updated_templates/valid", "valid.json")
