from dataclasses import dataclass

NCOLS = 28
NROWS = 32
NPIXELS = NCOLS * NROWS


@dataclass
class ChipParams:
    vcasb: int  # mV
    ireset: int  # pA (pixel)
    vcasn: int  # mV
    ibias: int  # nA (pixel)
    idb: int  # nA (pixel)
    vh: int  # mV


def get_run_id_inj(cp: ChipParams, col: int, row: int) -> str:
    return f"vcasb{cp.vcasb}_ireset{cp.ireset}_vcasn{cp.vcasn}_ibias{cp.ibias}_idb{cp.idb}_vh{cp.vh}_c{col}_r{row}_run_0"


def get_run_id_fhr(cp: ChipParams) -> str:
    return f"vcasb{cp.vcasb}_ireset{cp.ireset}_vcasn{cp.vcasn}_ibias{cp.ibias}_idb{cp.idb}_run_0"
