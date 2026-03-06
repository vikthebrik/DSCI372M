# Dataset

**Electrical Grid Stability Simulated Dataset**
UCI Machine Learning Repository — ID 471

URL: https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data

The notebook fetches this dataset automatically via:
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=471)
```

10,000 instances, 12 features (tau1–tau4, p1–p4, g1–g4), binary target: `stabf` (stable/unstable).
