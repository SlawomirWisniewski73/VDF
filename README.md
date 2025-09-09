# Vectorized Delta Format (VDF)

**Status:** Proof of Concept (MVP)  
**License:** AGPL-3.0-only

VDF to format zapisu dynamicznych danych, w którym kolejne stany są reprezentowane przez **wektory różnic (Δ)** zamiast pełnych snapshotów.

**Zalety:**
- oszczędność miejsca i IO,
- szybka rekonstrukcja stanów z checkpointów,
- proste API (Python, brak zależności),
- gotowe przykłady i benchmark.

---

## 🔧 Instalacja
Skopiuj `vdf.py` do swojego projektu lub użyj bezpośrednio w tym repo. Nie ma żadnych zewnętrznych zależności.

---

## 🚀 Quickstart

```python
from vdf import VDFStream, Header

S0 = [10, 10, 10, 10]
vdf = VDFStream(S0, header=Header(dimensions=[4], checkpoint_interval=2))

vdf.append_delta(indices=[2], values=[15], op="set")      # Δ1
vdf.append_delta(indices=[2, 3], values=[1, 2], op="add") # Δ2 -> checkpoint
vdf.append_delta(indices=[3], values=[2], op="mul")       # Δ3

print(vdf.get_state(0))  # [10, 10, 10, 10]
print(vdf.get_state(1))  # [10, 10, 15, 10]
print(vdf.get_state(2))  # [10, 10, 16, 12]
print(vdf.get_state(3))  # [10, 10, 16, 24]

```
---
## 📊 Mini-benchmark
```

python benchmark_minimal.py
Przykładowe wyniki (N=20 000, kroki=200, zmienność=0.1%):

=== VDF Mini-Benchmark ===
N (state length):         20,000
Steps (deltas):           200
Sparsity per step:        0.100% (~20 changes/step)
Build time VDF:           0.10 s
Approx FULL snapshots:    11.44 MB (JSON est.)
VDF stream size (JSON):   0.68 MB
Reconstruction timings:
   S_10:   ~5.9 ms
  S_200:   ~5.7 ms

```
---

## 🖼️ Demo wizualne
```

python examples/quickstart_visual.py            # zapisze PNG do examples/img/
python examples/quickstart_visual.py --show     # spróbuje wyświetlić okna
Obrazy trafią do examples/img/vdf_state_S0.png itd.

```
---
 
## 📜 Licencja

```
AGPL-3.0-only — szczegóły w pliku LICENSE.
Jeśli potrzebujesz licencji komercyjnej — napisz do autora.

```
---
