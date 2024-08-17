import json
import os


directory = "simulation/examples/job_seeking/configs"
meta_data_path = "all_seeker_agent_configs.json"

scale_map = {
    "2.5K": 1,
    "25K": 10,
    "250K": 100,
    "2.5M": 1000,
}


def main():
    data = json.load(open(os.path.join(directory, meta_data_path), "r"))
    new_data = data * max(scale_map.values())
    for prefix, r in scale_map.items():
        new_data = new_data[:len(data)*r]
        with open(os.path.join(directory, f"{prefix}_{meta_data_path}"), "w") as f:
            json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    main()