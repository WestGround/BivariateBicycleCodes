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


def _append_detector_block(
    out: stim.Circuit,
    *,
    current: Sequence[int],
    previous: Sequence[int | None],
    measurement_count: int,
) -> None:
    """Append detectors for one check type using absolute measurement indices."""
    if len(current) != len(previous):
        raise ValueError("Current and previous measurement blocks differ in length")
    for now, before in zip(current, previous):
        targets = [stim.target_rec(now - measurement_count)]
        if before is not None:
            targets.append(stim.target_rec(before - measurement_count))
        out.append("DETECTOR", targets)


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
    """Translate the scheduled cycle into a detector-annotated Stim circuit.

    The first ``num_cycles`` rounds are noisy and two trailing rounds are
    noiseless, matching the original decoder.  Detector order per cycle is
    Z-check measurements first (X-error syndrome), then X-check measurements
    (Z-error syndrome).
    """
    n2 = len(xchecks)
    if len(zchecks) != n2 or len(data_qubits) != 2 * n2:
        raise ValueError("Unexpected BB qubit partition")

    layers: list[list[Gate]] = []
    layer: list[Gate] = []
    for gate in cycle:
        if gate[0] == "TICK":
            layers.append(layer)
            layer = []
        else:
            layer.append(gate)
    if layer:
        raise ValueError("Scheduled cycle must end with TICK")
    if not layers:
        raise ValueError("Scheduled cycle contains no timesteps")

    data_indices = np.fromiter((lin_order[q] for q in data_qubits), dtype=np.intp)
    out = stim.Circuit()
    supplied = (hx is not None, hz is not None, lz is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("hx, hz, and lz must be supplied together")
    if all(supplied):
        assert hx is not None and hz is not None and lz is not None
        out += _encoded_state_prep(hx, hz, lz, data_indices)
        out.append("TICK")

    measurement_count = 0
    previous_z: list[int | None] = [None] * n2
    previous_x: list[int | None] = [None] * n2

    for cycle_index in range(num_cycles + 2):
        noisy = cycle_index < num_cycles
        init_p = noise.init if noisy else 0.0
        idle_p = noise.idle if noisy else 0.0
        cnot_p = noise.cnot if noisy else 0.0
        meas_p = noise.meas if noisy else 0.0

        current_z: list[int] | None = None
        current_x: list[int] | None = None

        for timestep, layer in enumerate(layers):
            by_kind: dict[str, list[Gate]] = {
                "PrepX": [],
                "PrepZ": [],
                "MeasZ": [],
                "MeasX": [],
                "CNOT": [],
                "IDLE": [],
            }
            for gate in layer:
                if gate[0] not in by_kind:
                    raise ValueError(f"Unsupported scheduled gate {gate[0]!r}")
                by_kind[gate[0]].append(gate)

            if by_kind["PrepX"]:
                targets = _targets(by_kind["PrepX"], lin_order)
                out.append("RX", targets)
                if init_p:
                    out.append("Z_ERROR", targets, init_p)
            if by_kind["PrepZ"]:
                targets = _targets(by_kind["PrepZ"], lin_order)
                out.append("R", targets)
                if init_p:
                    out.append("X_ERROR", targets, init_p)
            if by_kind["MeasZ"]:
                if current_z is not None:
                    raise ValueError("Each cycle may contain only one MeasZ layer")
                if len(by_kind["MeasZ"]) != n2:
                    raise ValueError(
                        f"MeasZ timestep {timestep} has {len(by_kind['MeasZ'])} "
                        f"measurements, expected {n2}"
                    )
                out.append("M", _targets(by_kind["MeasZ"], lin_order), meas_p)
                current_z = list(range(measurement_count, measurement_count + n2))
                measurement_count += n2
            if by_kind["MeasX"]:
                if current_x is not None:
                    raise ValueError("Each cycle may contain only one MeasX layer")
                if len(by_kind["MeasX"]) != n2:
                    raise ValueError(
                        f"MeasX timestep {timestep} has {len(by_kind['MeasX'])} "
                        f"measurements, expected {n2}"
                    )
                out.append("MX", _targets(by_kind["MeasX"], lin_order), meas_p)
                current_x = list(range(measurement_count, measurement_count + n2))
                measurement_count += n2
            if by_kind["CNOT"]:
                _append_cnot_layer(out, by_kind["CNOT"], lin_order, cnot_p)
            _append_idle_noise(out, by_kind["IDLE"], lin_order, idle_p)
            out.append("TICK")

        if current_z is None or current_x is None:
            raise ValueError("Each cycle must contain one MeasZ layer and one MeasX layer")
        # Keep detector order independent of the measurement timesteps:
        # Z-check detectors first, then X-check detectors.
        _append_detector_block(
            out,
            current=current_z,
            previous=previous_z,
            measurement_count=measurement_count,
        )
        _append_detector_block(
            out,
            current=current_x,
            previous=previous_x,
            measurement_count=measurement_count,
        )
        previous_z = list(current_z)
        previous_x = list(current_x)

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
