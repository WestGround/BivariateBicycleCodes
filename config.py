from dataclasses import dataclass
from pathlib import Path

from code_parameters import *


@dataclass(frozen=True)
class NoiseParameters:
    init: float
    idle: float
    cnot: float
    meas: float

    @classmethod
    def depolarizing(cls, p: float) -> "NoiseParameters":
        return cls(init=p, idle=p, cnot=p, meas=p)


@dataclass(frozen=True)
class BpOsdParameters:
    bp_method: str
    max_iter: int
    ms_scaling_factor: float
    schedule: str
    omp_thread_count: int
    osd_method: str
    osd_order: int


@dataclass(frozen=True)
class BeamSearchParameters:
    max_rounds: int
    beam_width: int
    num_results: int
    initial_iters: int
    iters_per_round: int


@dataclass(frozen=True)
class OnlineParameters:
    num_trials: int
    batch_size: int
    random_seed: int | None
    progress: bool

# Number of QEC cycles repeated
N_c = 12

error_rate = 0.001
noise = NoiseParameters.depolarizing(error_rate)

# Select the code used by both offline and online scripts.
code_param = GrossParam
code_distance = 12

# Select the recovery decoder used by decoder_online.py.
# Supported values: "bp_osd", "beam_search".
decoder_backend = "beam_search"

# BP-OSD parameters use the ldpc v2 spelling.
bp_osd = BpOsdParameters(
    bp_method="minimum_sum",
    max_iter=30,
    ms_scaling_factor=1.0,
    schedule="parallel",
    omp_thread_count=1,
    osd_method="osd_cs",
    osd_order=10,
)

beam_search = BeamSearchParameters(
    max_rounds=20,
    beam_width=64,
    num_results=1,
    initial_iters=40,
    iters_per_round=30,
)

online = OnlineParameters(
    num_trials=500_000,
    batch_size=4_096,
    random_seed=None,
    progress=True,
)

decoder_data_dir = Path("TMP")
result_dir = Path("results")
