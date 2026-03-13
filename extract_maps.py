import json

import numpy as np

with open("instances_data/instances_connected_room.json") as f:
    data = json.load(f)

last_obs_p = None
last_obs_r = None
ndiff = 0
for item in data:
    obs_p = np.array(item["obstacles"]["positions"])
    obs_r = np.array(item["obstacles"]["radii"])

    if last_obs_p is None:
        last_obs_p = obs_p
        last_obs_r = obs_r
        continue

    if not np.all(obs_p == last_obs_p):
        print(np.max(np.abs(obs_p - last_obs_p)))
        print("different")
        last_obs_p = obs_p
        last_obs_r = obs_r
        ndiff += 1

print("num diff:", ndiff)
