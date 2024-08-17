import json
import os
import numpy as np
from copy import deepcopy


directory = "simulation/examples/recommendation/configs"
meta_data_path = "all_recuser_agent_configs.json"

scale_map = {
    "1K": 1,
    "10K": 10,
    "100K": 100,
    "1M": 1000,
}


def main():
    data = json.load(open(os.path.join(directory, meta_data_path), "r"))
    base_num = len(data)
    cnt = len(data)
    max_rate = max(scale_map.values())
    new_data = []
    for i in range(max_rate):
        new_data.extend(deepcopy(data))
    for i in range(1, max_rate):
        cur_data = new_data[i*base_num:(i+1)*base_num]
        for d in cur_data:
            tmp_d = np.array(d["args"]["relationship"]) + cnt
            d["args"]["relationship"] = tmp_d.tolist()
        new_data[i*base_num:(i+1)*base_num] = cur_data
        cnt += base_num

    for prefix, r in scale_map.items():
        with open(os.path.join(directory, f"{prefix}_{meta_data_path}"), "w") as f:
            json.dump(new_data[:base_num*r], f, indent=4)

if __name__ == "__main__":
    main()