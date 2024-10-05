import json
import os


directory = "simulation/examples/job_seeking/configs"
meta_data_path = "all_SeekerAgent_configs.json"

scale_map = {
    "2421": 1,
    "9684": 4,
    "24210": 10,
    "48420": 20,
    "99261": 41,
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