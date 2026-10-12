import time

from constellation.core.controller import ScriptableController
from constellation.core.controller_configuration import load_config
from constellation.core.protocol.cscp1 import SatelliteState

config_file_path = "/home/teleuser/Nellie/SPARC.toml"
group_name = "sparc"
n_injections = 25
pulse_interval = 0.001  #s between two pulses on the same pixel

NROWS = 32
NCOLS = 28
N_COL_REGISTERS = 15

ENABLED_COL_REGISTERS = range(0, 15)


def col_register(col: int) -> str:
    return f"enable_col_{(col + 1) // 2}"


def row_register(row: int, col: int) -> str:
    even_odd = "even" if col % 2 == 0 else "odd"
    return f"enable_row_{row}_{even_odd}"


def set_register(name: str, value: int) -> None:
    ctrl.constellation.Caribou.SPARC._set_register(["sparc", name, value])


def disable_all_pixels() -> None:
    for n in range(N_COL_REGISTERS):
        set_register(f"enable_col_{n}", 0)
    for row in range(NROWS):
        set_register(f"enable_row_{row}_even", 0)
        set_register(f"enable_row_{row}_odd", 0)


def restore_mask() -> None:
    """Put the enable registers back to what SPARC.toml sets"""
    for n in range(N_COL_REGISTERS):
        set_register(f"enable_col_{n}", 1 if n in ENABLED_COL_REGISTERS else 0)
    for row in range(NROWS):
        set_register(f"enable_row_{row}_even", 1)
        set_register(f"enable_row_{row}_odd", 1)


pixels = [(row, col) for col in range(NCOLS) if (col + 1) // 2 in ENABLED_COL_REGISTERS for row in range(NROWS)]

ctrl = ScriptableController(group_name)
cfg = load_config(config_file_path)
ctrl.await_satellites(["MessagePackWriter.SPARC", "Keithley.Bias", "Caribou.SPARC"])

ctrl.constellation.initialize(cfg)
ctrl.await_state(SatelliteState.INIT, timeout=5)

ctrl.constellation.launch()
ctrl.await_state(SatelliteState.ORBIT, timeout=30)

for vh in range(95, 160, 5):
    ctrl.log.status(f"Reconfuring Vh to {vh}mV")

    recfg = {"devices": {"SPARC": {"vh": vh * 1e-3}}}
    ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
    ctrl.await_state(SatelliteState.ORBIT)

    run_identifier = f"run_vh_{vh}"
    ctrl.log.status(f"Starting run {run_identifier}")
    ctrl.constellation.start(run_identifier)
    ctrl.await_state(SatelliteState.RUN)

    ctrl.constellation.Caribou.SPARC._set_memory(["sparc", "pulse_duration", 1000])

    disable_all_pixels()
    for row, col in pixels:
        set_register(col_register(col), 1)
        set_register(row_register(row, col), 1)

        for n in range(n_injections):
            ctrl.constellation.Caribou.SPARC._set_memory(["sparc", "pulse_cmd", 1])
            time.sleep(pulse_interval)

        # Give the last hit time to be read out before masking the pixel again
        time.sleep(pulse_interval)
        set_register(col_register(col), 0)
        set_register(row_register(row, col), 0)

    # Put the mask back so the next run starts with the same settings
    restore_mask()

    # Stop the run and await ORBIT state of all satellites
    ctrl.constellation.stop()
    ctrl.await_state(SatelliteState.ORBIT)


# Land
ctrl.constellation.land()
ctrl.await_state(SatelliteState.INIT)
