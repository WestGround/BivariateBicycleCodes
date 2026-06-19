from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BBParameters:
    ell: int
    m: int
    a1: int
    a2: int
    a3: int
    b1: int
    b2: int
    b3: int


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
class OnlineParameters:
    num_trials: int
    batch_size: int
    random_seed: int | None
    progress: bool


# [[144, 12, 12]]
GrossParam = BBParameters(
    ell = 12,
    m = 6,
    a1 = 3,
    a2 = 1,
    a3 = 2,
    b1 = 3,
    b2 = 1,
    b3 = 2
)

# [[72, 12, 6]]
HalfGrossParam = BBParameters(
    ell = 6,
    m = 6,
    a1 = 3,
    a2 = 1,
    a3 = 2,
    b1 = 3,
    b2 = 1,
    b3 = 2
)

# [[288, 12, 18]]
TwoGrossParam = BBParameters(
    ell = 12,
    m = 12,
    a1 = 3,
    a2 = 2,
    a3 = 7,
    b1 = 3,
    b2 = 1,
    b3 = 2
)

sched_x: list[int | str] = ["idle", 1, 4, 3, 5, 0, 2]
sched_z: list[int | str] = [3, 5, 0, 1, 2, 4, "idle"]
N_c = 12

error_rate = 0.001
noise = NoiseParameters.depolarizing(error_rate)

# Select the code used by both offline and online scripts.
code_param = GrossParam
code_distance = 12

# BP-OSD parameters use the ldpc v2 spelling.
bp_osd = BpOsdParameters(
    bp_method="minimum_sum",
    max_iter=10_000,
    ms_scaling_factor=1.0,
    schedule="parallel",
    omp_thread_count=1,
    osd_method="osd_cs",
    osd_order=7,
)

online = OnlineParameters(
    num_trials=10_000,
    batch_size=2_048,
    random_seed=None,
    progress=True,
)

decoder_data_dir = Path("TMP")
result_dir = Path("results")
