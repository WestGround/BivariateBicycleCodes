"""Stim circuit construction and fast error-frame sampling for BB codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import stim

from config import NoiseParameters


Gate = tuple


@dataclass(frozen=True)
class StimCircuitData:
    circuit: stim.Circuit
    data_indices: np.ndarray
    xcheck_indices: np.ndarray
    zcheck_indices: np.ndarray
    n2: int
    num_cycles: int


def _targets(gates: Sequence[Gate], lin_order: dict) -> list[int]:
    return [lin_order[gate[1]] for gate in gates]


def _take(
    cycle: Sequence[Gate], position: int, count: int, kind: str
) -> tuple[list[Gate], int]:
    gates = list(cycle[position : position + count])
    if len(gates) != count or any(gate[0] != kind for gate in gates):
        actual = tuple(gate[0] for gate in gates[:3])
        raise ValueError(
            f"Unexpected cycle layout at {position}: expected {kind}, got {actual}"
        )
    return gates, position + count


def _append_cnot_layer(
    out: stim.Circuit,
    gates: Sequence[Gate],
    lin_order: dict,
    probability: float,
) -> None:
    pairs: list[int] = []
    used: set[int] = set()
    for gate in gates:
        control = lin_order[gate[1]]
        target = lin_order[gate[2]]
        if control in used or target in used:
            raise ValueError("CNOT layer is not parallel")
        used.add(control)
        used.add(target)
        pairs.extend((control, target))
    out.append("CX", pairs)
    if probability:
        out.append("DEPOLARIZE2", pairs, probability)


def _append_idle_noise(
    out: stim.Circuit,
    gates: Sequence[Gate],
    lin_order: dict,
    probability: float,
) -> None:
    if probability:
        out.append("DEPOLARIZE1", _targets(gates, lin_order), probability)


def _append_detectors(out: stim.Circuit, n2: int, cycle_index: int) -> None:
    """Append current syndrome xor previous syndrome, in check-index order."""
    for check in range(n2):
        current = stim.target_rec(check - n2)
        if cycle_index == 0:
            out.append("DETECTOR", [current])
        else:
            previous = stim.target_rec(check - 3 * n2)
            out.append("DETECTOR", [current, previous])


def _encoded_state_prep(
    hx: np.ndarray,
    hz: np.ndarray,
    lz: np.ndarray,
    data_indices: np.ndarray,
) -> stim.Circuit:
    """Synthesize a noiseless canonical logical-|0> encoded state."""
    hx = np.asarray(hx, dtype=np.uint8)
    hz = np.asarray(hz, dtype=np.uint8)
    lz = np.asarray(lz, dtype=np.uint8)
    n = hx.shape[1]
    if hz.shape[1] != n or lz.shape[1] != n or len(data_indices) != n:
        raise ValueError("PCM/logical dimensions do not match the data qubits")

    stabilizers: list[stim.PauliString] = []
    for row in hx:
        stabilizers.append(stim.PauliString("".join("X" if bit else "_" for bit in row)))
    for row in hz:
        stabilizers.append(stim.PauliString("".join("Z" if bit else "_" for bit in row)))
    # Logical Zs fix a canonical logical |0...0> state.
    for row in lz:
        stabilizers.append(stim.PauliString("".join("Z" if bit else "_" for bit in row)))
    tableau = stim.Tableau.from_stabilizers(stabilizers, allow_redundant=True)
    local = tableau.to_circuit("graph_state")

    remapped = stim.Circuit()
    for instruction in local:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Unexpected repeat block in encoded-state preparation")
        targets: list[int] = []
        for target in instruction.targets_copy():
            if not target.is_qubit_target:
                raise ValueError("Unexpected non-qubit target in encoded-state preparation")
            targets.append(int(data_indices[target.value]))
        remapped.append(
            instruction.name,
            targets,
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
    return remapped


def build_stim_circuit(
    *,
    cycle: Sequence[Gate],
    lin_order: dict,
    data_qubits: Sequence[tuple],
    xchecks: Sequence[tuple],
    zchecks: Sequence[tuple],
    num_cycles: int,
    noise: NoiseParameters,
    hx: np.ndarray | None = None,
    hz: np.ndarray | None = None,
    lz: np.ndarray | None = None,
) -> StimCircuitData:
    """Translate the legacy scheduled cycle into a detector-annotated circuit.

    The first ``num_cycles`` rounds are noisy and two trailing rounds are
    noiseless, matching the original decoder.  Detector order per cycle is
    Z-check measurements first (X-error syndrome), then X-check measurements
    (Z-error syndrome).
    """
    n2 = len(xchecks)
    if len(zchecks) != n2 or len(data_qubits) != 2 * n2:
        raise ValueError("Unexpected BB qubit partition")

    position = 0
    prep_x, position = _take(cycle, position, n2, "PrepX")
    round0, position = _take(cycle, position, n2, "CNOT")
    idle0, position = _take(cycle, position, n2, "IDLE")
    middle: list[list[Gate]] = []
    for _ in range(5):
        xgates, position = _take(cycle, position, n2, "CNOT")
        zgates, position = _take(cycle, position, n2, "CNOT")
        middle.append(xgates + zgates)
    meas_z, position = _take(cycle, position, n2, "MeasZ")
    round6, position = _take(cycle, position, n2, "CNOT")
    idle6, position = _take(cycle, position, n2, "IDLE")
    idle7, position = _take(cycle, position, 2 * n2, "IDLE")
    meas_x, position = _take(cycle, position, n2, "MeasX")
    prep_z, position = _take(cycle, position, n2, "PrepZ")
    if position != len(cycle):
        raise ValueError(f"Cycle has {len(cycle) - position} unparsed operations")

    data_indices = np.fromiter((lin_order[q] for q in data_qubits), dtype=np.intp)
    prep_x_q = _targets(prep_x, lin_order)
    prep_z_q = _targets(prep_z, lin_order)
    meas_z_q = _targets(meas_z, lin_order)
    meas_x_q = _targets(meas_x, lin_order)
    out = stim.Circuit()
    supplied = (hx is not None, hz is not None, lz is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("hx, hz, and lz must be supplied together")
    if all(supplied):
        assert hx is not None and hz is not None and lz is not None
        out += _encoded_state_prep(hx, hz, lz, data_indices)
        out.append("TICK")

    for cycle_index in range(num_cycles + 2):
        noisy = cycle_index < num_cycles
        init_p = noise.init if noisy else 0.0
        idle_p = noise.idle if noisy else 0.0
        cnot_p = noise.cnot if noisy else 0.0
        meas_p = noise.meas if noisy else 0.0

        # t=0: X-check preparation, Z-check interaction, unused data idles.
        out.append("RX", prep_x_q)
        if init_p:
            out.append("Z_ERROR", prep_x_q, init_p)
        _append_cnot_layer(out, round0, lin_order, cnot_p)
        _append_idle_noise(out, idle0, lin_order, idle_p)
        out.append("TICK")

        # t=1..5: disjoint X- and Z-check interactions.
        for layer in middle:
            _append_cnot_layer(out, layer, lin_order, cnot_p)
            out.append("TICK")

        # t=6: Z-check measurement and final X-check interaction.
        out.append("M", meas_z_q, meas_p)
        _append_detectors(out, n2, cycle_index)
        _append_cnot_layer(out, round6, lin_order, cnot_p)
        _append_idle_noise(out, idle6, lin_order, idle_p)
        out.append("TICK")

        # t=7: data idles, X checks are measured, Z checks are prepared.
        _append_idle_noise(out, idle7, lin_order, idle_p)
        out.append("MX", meas_x_q, meas_p)
        _append_detectors(out, n2, cycle_index)
        out.append("R", prep_z_q)
        if init_p:
            out.append("X_ERROR", prep_z_q, init_p)
        out.append("TICK")

    expected_detectors = 2 * n2 * (num_cycles + 2)
    if out.num_detectors != expected_detectors:
        raise AssertionError(
            f"Built {out.num_detectors} detectors, expected {expected_detectors}"
        )
    return StimCircuitData(
        circuit=out,
        data_indices=data_indices,
        xcheck_indices=np.fromiter((lin_order[q] for q in xchecks), dtype=np.intp),
        zcheck_indices=np.fromiter((lin_order[q] for q in zchecks), dtype=np.intp),
        n2=n2,
        num_cycles=num_cycles,
    )


class StimBatchSimulator:
    """Reusable batched Stim error-frame simulator."""

    def __init__(
        self,
        circuit_data: StimCircuitData,
        *,
        batch_size: int,
        seed: int | None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.data = circuit_data
        self.batch_size = batch_size
        self.simulator = stim.FlipSimulator(
            batch_size=batch_size,
            num_qubits=max(circuit_data.circuit.num_qubits, 4 * circuit_data.n2),
            disable_stabilizer_randomization=True,
            seed=seed,
        )

    def sample(
        self, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not 0 < count <= self.batch_size:
            raise ValueError("count must be in [1, batch_size]")
        self.simulator.clear()
        self.simulator.do(self.data.circuit)
        xs, zs, _, detectors, _ = self.simulator.to_numpy(
            transpose=True,
            output_xs=True,
            output_zs=True,
            output_detector_flips=True,
        )
        assert xs is not None and zs is not None and detectors is not None
        n2 = self.data.n2
        total_cycles = self.data.num_cycles + 2
        detectors = detectors[:count].reshape(count, total_cycles, 2 * n2)
        syndrome_x = detectors[:, :, :n2].reshape(count, -1)
        syndrome_z = detectors[:, :, n2:].reshape(count, -1)
        return (
            syndrome_x.astype(np.uint8, copy=False),
            syndrome_z.astype(np.uint8, copy=False),
            xs[:count, self.data.data_indices].astype(np.uint8, copy=False),
            zs[:count, self.data.data_indices].astype(np.uint8, copy=False),
        )
