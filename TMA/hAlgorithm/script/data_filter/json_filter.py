import argparse
import csv
import json


def filter_sf_json(json_data, csv_data, csv_header):
    filtered_idx = []
    for data in csv_data:
        data_idx = int(data[csv_header.index("data_idx")])
        filtered_idx.append(data_idx)
    filter_data = list(filter(lambda x: x[0] not in filtered_idx, enumerate(json_data["files"])))
    filter_data = [data[1] for data in filter_data]
    print(f'json_data_len: {len(json_data["files"])}; filtered_json_data_len: {len(filter_data)}.')
    return {"files": filter_data}


def filter_mf_json(json_data, csv_data, csv_header):
    filter_scene = []
    for data in csv_data:
        data_scene = data[csv_header.index("scene")]
        if data_scene not in filter_scene:
            filter_scene.append(data_scene)
    filter_data = {k: v for k, v in json_data["mf_files"].items() if k not in filter_scene}
    print(
        f'json_data_len: {len(json_data["mf_files"])}; filtered_json_data_len: {len(filter_data)}.'
    )
    return {"mf_files": filter_data}


def filter_json(json_path, csv_path):
    # Read JSON file
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    # Read CSV file
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_header = next(csv_reader)  # Extract the header from the CSV file

        # Check if 'data_index' exists in the header
        if "data_idx" not in csv_header:
            raise ValueError("The CSV header does not contain 'data_index'.")

        csv_data = list(csv_reader)

    print(f"Filtering JSON data from {json_path}...")

    if "files" in json_data:
        filter_data = filter_sf_json(json_data, csv_data, csv_header)
    elif "mf_files" in json_data:
        filter_data = filter_mf_json(json_data, csv_data, csv_header)
    else:
        raise ValueError("The JSON file does not contain 'files' or 'mf_files'.")

    data = csv_data[0][csv_header.index("rgb_path")].split("/")[0]
    import os

    output_json_path = (
        "/mnt/netdata/Team/AI/datasets/TMD/FilterLoss/"
        + str(data)
        + "_"
        + os.path.basename(json_path.replace(".json", "_filter_loss.json"))
    )
    output_json_path = os.path.join(
        os.path.dirname(csv_path),
        str(data) + "_" + os.path.basename(json_path.replace(".json", "_filter_loss.json")),
    )
    print(f"Saving filter JSON data to {output_json_path}...")
    with open(output_json_path, "w") as json_file:
        json.dump(filter_data, json_file, indent=2)


def main(json_path, csv_path):
    # Call filter_json with the provided arguments
    filter_json(json_path, csv_path)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Filter JSON data from loss CSV.")
    parser.add_argument("--json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")

    # Parse arguments
    args = parser.parse_args()
    main(args.json, args.csv)
