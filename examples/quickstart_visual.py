# Demo działania VDF z prostą wizualizacją kroków.
# Zapisuje PNG do examples/img/ lub pokazuje okna przy flagze --show.

import os
import sys
import copy
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

# Matplotlib w trybie headless, jeśli brak GUI
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

Op = Literal["set", "add", "mul"]

@dataclass
class Header:
    version: str = "0.1"
    datatype: str = "int32"
    dimensions: List[int] = None
    checkpoint_interval: Optional[int] = 2

@dataclass
class DeltaFrame:
    indices: List[int]
    values: List[int]
    op: Op = "set"

@dataclass
class CheckpointFrame:
    full_state: List[int]

Frame = Union[DeltaFrame, CheckpointFrame]

class VDFStream:
    def __init__(self, S0: List[int], header: Optional[Header] = None):
        if header is None:
            header = Header(dimensions=[len(S0)])
        if header.dimensions is None:
            header.dimensions = [len(S0)]
        self.header = header
        self.S0 = copy.deepcopy(S0)
        self.frames: List[Frame] = []

    def append_delta(self, indices: List[int], values: List[int], op: Op = "set"):
        self.frames.append(DeltaFrame(indices=list(indices), values=list(values), op=op))

    def _apply(self, S: List[int], delta: DeltaFrame):
        if delta.op == "set":
            for i, v in zip(delta.indices, delta.values):
                S[i] = v
        elif delta.op == "add":
            for i, v in zip(delta.indices, delta.values):
                S[i] = S[i] + v
        elif delta.op == "mul":
            for i, v in zip(delta.indices, delta.values):
                S[i] = S[i] * v

    def get_state(self, k: int) -> List[int]:
        if k == 0:
            return copy.deepcopy(self.S0)
        S = copy.deepcopy(self.S0)
        steps = 0
        for f in self.frames:
            if isinstance(f, DeltaFrame):
                steps += 1
                self._apply(S, f)
                if steps == k:
                    break
        return S

# Stała, przewidywalna lokalizacja na obrazy w repo

def _resolve_out_dir() -> str:
    return os.path.join(os.getcwd(), "examples", "img")


def main(show: bool = False):
    # Stan początkowy (8 elementów)
    S0 = [0, 0, 0, 0, 0, 0, 0, 0]
    vdf = VDFStream(S0)

    # 3 delty
    vdf.append_delta([1], [5], op="set")        # Δ1
    vdf.append_delta([2, 6], [3, 2], op="add")  # Δ2
    vdf.append_delta([3, 6], [2, 3], op="mul")  # Δ3

    # Testy akceptacyjne (nie ruszać — w razie zmian logiki zaktualizować oczekiwania)
    expected_states = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 0, 0],
        [0, 5, 3, 0, 0, 0, 2, 0],
        [0, 5, 3, 0, 0, 0, 6, 0],
    ]
    for k, exp in zip(range(4), expected_states):
        assert vdf.get_state(k) == exp, f"Nieoczekiwany stan S_{k}: {vdf.get_state(k)} != {exp}"

    # Rekonstrukcja i rysowanie
    states = [vdf.get_state(k) for k in range(4)]
    out_dir = _resolve_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    for k, state in enumerate(states):
        plt.figure(figsize=(6, 3))
        plt.bar(range(len(state)), state)
        plt.title(f"VDF state S_{k}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            out_path = os.path.join(out_dir, f"vdf_state_S{k}.png")
            plt.savefig(out_path, dpi=120)
            print(f"Saved: {out_path}")
        plt.close()

if __name__ == "__main__":
    show_flag = "--show" in sys.argv
    main(show=show_flag)
