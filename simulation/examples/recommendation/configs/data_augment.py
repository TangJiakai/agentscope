import json
import math
import os
import sys
import random
from copy import deepcopy

directory = "simulation/examples/recommendation/configs"
meta_data_path = "all_RecUserAgent_configs.json"

scale_map = {
    "3139": 0.015625,
    "6279": 0.03125,
    "12559": 0.0625,
    "25118": 0.125,
    "50237": 0.25,
    "100474": 0.5,
    "200948": 1,
}


def main():
    data = json.load(open(os.path.join(directory, meta_data_path), "r"))
    origin_data_sz = len(data)
    print(f"The original datasize is {origin_data_sz}")
    new_data = []
    for v in scale_map.values():
        new_data_sz = math.floor(v * origin_data_sz)
        new_data_part = deepcopy(data[:new_data_sz])
        for i, d in enumerate(new_data_part):
            d['args']['relationship'] = random.sample(
                list(range(i)) + list(range(i + 1, new_data_sz)),
                5
            )
        new_data.append(new_data_part)
        print(f"generated datasize is {math.floor(v * origin_data_sz)}")

    for prefix, d in zip(scale_map.keys(), new_data):
        with open(os.path.join(directory, f"{prefix}_{meta_data_path}"), "w") as f:
            json.dump(d, f, indent=4)
        print(f"Generated {prefix}_{meta_data_path}")

if __name__ == "__main__":
    main()