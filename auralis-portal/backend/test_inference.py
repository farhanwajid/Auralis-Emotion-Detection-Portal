import time, json
from auralis_core import load_auralis, predict

t = time.time()
model, cfg = load_auralis("model")
print(f"Model loaded in : {time.time() - t:.1f} seconds")

for i in range(3):
    t = time.time()
    out = predict(model, cfg, "samples/03-01-01-01-01-01-01.wav")
    print(f"run {i+1}: {time.time() - t:.2f}s")

print(out["emotion"], round(out["confidence"], 3))
print(json.dumps(out["probabilities"], indent=2))