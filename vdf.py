```python
# SPDX-License-Identifier: AGPL-3.0-only
# Vectorized Delta Format (VDF) - Minimal Proof of Concept (MVP)

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Dict, Any, Iterable, Union
import json
import copy

Op = Literal["set", "add", "mul"]

@dataclass
class Header:
    version: str = "0.1"
    datatype: str = "float64"
    dimensions: List[int] = None
    checkpoint_interval: Optional[int] = None

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
    """
    Minimalna referencyjna implementacja VDF dla wektorów 1D (spłaszczone tensory).
    Brak zależności zewnętrznych. Prosta serializacja JSON.
    """
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

    def iter_states(self) -> Iterable[List[Any]]:
        yield copy.deepcopy(self.S0)
        S = copy.deepcopy(self.S0)
        for f in self.frames:
            if isinstance(f, DeltaFrame):
                self._apply(S, f)
                yield copy.deepcopy(S)

    def to_json(self) -> str:
        def frame_to_dict(f: Frame) -> Dict[str, Any]:
            if isinstance(f, DeltaFrame):
                return {"type": "delta", "indices": f.indices, "values": f.values, "op": f.op}
            else:
                return {"type": "checkpoint", "full_state": f.full_state}
        data = {
            "header": asdict(self.header),
            "S0": self.S0,
            "frames": [frame_to_dict(f) for f in self.frames],
        }
        return json.dumps(data)

    @staticmethod
    def from_json(s: str) -> "VDFStream":
        o = json.loads(s)
        header = Header(**o["header"])
        stream = VDFStream(S0=o["S0"], header=header)
        for f in o["frames"]:
            if f["type"] == "delta":
                stream.frames.append(DeltaFrame(indices=f["indices"], values=f["values"], op=f["op"]))
            elif f["type"] == "checkpoint":
                stream.frames.append(CheckpointFrame(full_state=f["full_state"]))
            else:
                raise ValueError("Unknown frame type")
        return stream
