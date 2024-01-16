from pathlib import Path

import yaml

map_files = ["mazes-maps", "random-pico"]

maps = {}
for file_name in map_files:
    with open(Path(__file__).parent / f"{file_name}.yaml", "r") as f:
        maps.update(**yaml.safe_load(f))

MAPS_REGISTRY = maps
