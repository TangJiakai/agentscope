import json
import os


directory = "simulation/examples/job_seeking/configs"
meta_data_path = "all_seeker_agent_configs.json"

scale_map = {
    "3234": 1,
    "12918": 5,
    "25023": 10,
    "97653": 40,
    "121863": 50,
    "1211313": 500,
}


def main():
    data = json.load(open(os.path.join(directory, meta_data_path), "r"))
    new_data = data * max(scale_map.values())
    for prefix, r in scale_map.items():
        tmp_data = new_data[:len(data)*r]
        with open(os.path.join(directory, f"{prefix}_{meta_data_path}"), "w") as f:
            json.dump(tmp_data, f, indent=4)

if __name__ == "__main__":
    main()