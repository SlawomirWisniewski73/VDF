# Mini-benchmark VDF: porównanie rozmiaru i czasu rekonstrukcji.
# Uruchom:  python benchmark_minimal.py

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Dict, Any, Iterable, Union
import json, copy, random, time

Op = Literal["set", "add", "mul"]

@dataclass
class Header:
    version: str = "0.1"
    datatype: str = "int32"
    dimensions: List[int] = None
    checkpoint_interval: Optional[int] = 20

@dataclass
class DeltaFrame:
    indices: List[int]
    values: List[Any]
    op: Op = "set"

@dataclass
class CheckpointFrame:
    full_state: List[Any]

Frame = Union[DeltaFrame, CheckpointFrame]

class VDFStream:
    def __init__(self, S0: List[Any], header: Optional[Header] = None):
        if header is None:
            header = Header(dimensions=[len(S0)])
        if header.dimensions is None:
            header.dimensions = [len(S0)]
        self.header = header
        self.S0 = copy.deepcopy(S0)
        self.frames: List[Frame] = []
        if self.header.checkpoint_interval is not None and self.header.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when provided")

    def append_delta(self, indices: List[int], values: List[Any], op: Op = "set"):
        if len(indices) != len(values):
            raise ValueError("indices and values length mismatch")
        self.frames.append(DeltaFrame(indices=list(indices), values=list(values), op=op))
        ci = self.header.checkpoint_interval
        if ci:
            applied_deltas = sum(isinstance(f, DeltaFrame) for f in self.frames)
            if applied_deltas % ci == 0:
                S = self.get_state(applied_deltas)
                self.frames.append(CheckpointFrame(full_state=S))

    def _apply(self, S: List[Any], delta: DeltaFrame):
        if delta.op == "set":
            for i, v in zip(delta.indices, delta.values):
                S[i] = v
        elif delta.op == "add":
            for i, v in zip(delta.indices, delta.values):
                S[i] = S[i] + v
        elif delta.op == "mul":
            for i, v in zip(delta.indices, delta.values):
                S[i] = S[i] * v
        else:
            raise ValueError(f"Unsupported op: {delta.op}")

    def _locate_checkpoint(self, k: int):
        steps = 0
        last_cp = None
        for f in self.frames:
            if isinstance(f, DeltaFrame):
                steps += 1
                if steps > k:
                    break
            else:
                last_cp = (steps, f.full_state)
        return last_cp

    def get_state(self, k: int) -> List[Any]:
        if k < 0:
            raise ValueError("k must be >= 0")
        if k == 0:
            return copy.deepcopy(self.S0)
        cp = self._locate_checkpoint(k)
        if cp is None:
            S = copy.deepcopy(self.S0)
            start = 1
        else:
            k_cp, S_cp = cp
            S = copy.deepcopy(S_cp)
            start = k_cp + 1
        steps = 0
        for f in self.frames:
            if isinstance(f, DeltaFrame):
                steps += 1
                if steps < start:
                    continue
                if steps > k:
                    break
                self._apply(S, f)
        return S

    def to_json(self) -> str:
        def frame_to_dict(f: Frame) -> Dict[str, Any]:
            if isinstance(f, DeltaFrame):
                return {"type": "delta", "indices": f.indices, "values": f.values, "op": f.op}
            else:
                return {"type": "checkpoint", "full_state": f.full_state}
        data = {"header": asdict(self.header), "S0": self.S0, "frames": [frame_to_dict(f) for f in self.frames]}
        return json.dumps(data)


def run_benchmark(N=20000, STEPS=200, SPARSITY=0.001, seed=42):
    random.seed(seed)
    S0 = [0] * N
    vdf = VDFStream(S0, header=Header(dimensions=[N], checkpoint_interval=20))
    changes_per_step = max(1, int(N * SPARSITY))

    t0 = time.perf_counter()
    for _ in range(STEPS):
        idxs = random.sample(range(N), changes_per_step)
        vals = [random.randint(1, 9) for _ in range(changes_per_step)]
        vdf.append_delta(indices=idxs, values=vals, op="add")
    t_build_vdf = time.perf_counter() - t0

    state_json_once = json.dumps(S0)
    approx_full_snapshots_bytes = len(state_json_once) * STEPS
    vdf_bytes = len(vdf.to_json())

    ks = sorted(random.sample(range(0, STEPS + 1), k=10))
    rec_times = []
    for k in ks:
        t1 = time.perf_counter()
        _ = vdf.get_state(k)
        rec_times.append(time.perf_counter() - t1)

    return {
        "N": N,
        "STEPS": STEPS,
        "SPARSITY": SPARSITY,
        "changes_per_step": changes_per_step,
        "build_time_vdf_s": t_build_vdf,
        "approx_full_snapshots_MB": approx_full_snapshots_bytes / (1024 * 1024),
        "vdf_json_MB": vdf_bytes / (1024 * 1024),
        "reconstruction_checks": ks,
        "reconstruction_times_ms": [t * 1000 for t in rec_times],
    }

if __name__ == "__main__":
    result = run_benchmark()
    print("=== VDF Mini-Benchmark ===")
    print(f"N (state length):         {result['N']:,}")
    print(f"Steps (deltas):           {result['STEPS']:,}")
    print(f"Sparsity per step:        {result['SPARSITY']*100:.3f}% (~{result['changes_per_step']} changes/step)")
    print(f"Build time VDF:           {result['build_time_vdf_s']:.3f} s")
    print(f"Approx FULL snapshots:    {result['approx_full_snapshots_MB']:.2f} MB (JSON est.)")
    print(f"VDF stream size (JSON):   {result['vdf_json_MB']:.2f} MB")
    print("Reconstruction timings:")
    for k, t in zip(result['reconstruction_checks'], result['reconstruction_times_ms']):
        print(f"  S_{k:>3}: {t:6.2f} ms")
