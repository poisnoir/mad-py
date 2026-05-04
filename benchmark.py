import timeit
import json
import pickle
import dataclasses
from mad import Mad, MadType

# 1. Define a complex nested structure to stress-test the library
@dataclasses.dataclass
class Vec3:
    x: MadType.float32
    y: MadType.float32
    z: MadType.float32

@dataclasses.dataclass
class PlayerData:
    id: MadType.uint64
    name: MadType.string
    pos: Vec3
    health: MadType.uint8

# 2. Prepare data
m = Mad(PlayerData)
p_obj = PlayerData(
    id=123456789, 
    name="Adventurer_01", 
    pos=Vec3(10.5, 20.1, -5.0), 
    health=100
)

# Equivalent dictionary for JSON/Pickle
p_dict = {
    "id": 123456789,
    "name": "Adventurer_01",
    "pos": {"x": 10.5, "y": 20.1, "z": -5.0},
    "health": 100
}

# 3. Size Comparison
mad_payload = m.encode(p_obj)
json_payload = json.dumps(p_dict).encode()
pickle_payload = pickle.dumps(p_obj)

print("--- Size Comparison (Bytes) ---")
print(f"Mad:    {len(mad_payload)} bytes")
print(f"JSON:   {len(json_payload)} bytes")
print(f"Pickle: {len(pickle_payload)} bytes")
print("-" * 30)

# 4. Speed Comparison
def run_mad():
    m.decode(m.encode(p_obj))

def run_json():
    json.loads(json.dumps(p_dict))

def run_pickle():
    pickle.loads(pickle.dumps(p_obj))

iters = 50_000
mad_time = timeit.timeit(run_mad, number=iters)
json_time = timeit.timeit(run_json, number=iters)
pickle_time = timeit.timeit(run_pickle, number=iters)

print(f"--- Speed Comparison ({iters:,} round-trips) ---")
print(f"Mad:    {mad_time:.4f}s ({iters/mad_time:.0f} ops/sec)")
print(f"JSON:   {json_time:.4f}s ({iters/json_time:.0f} ops/sec)")
print(f"Pickle: {pickle_time:.4f}s ({iters/pickle_time:.0f} ops/sec)")
