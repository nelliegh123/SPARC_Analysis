import time

from constellation.core.controller import ScriptableController
from constellation.core.controller_configuration import load_config
from constellation.core.protocol.cscp1 import SatelliteState

# Settings
config_file_path = "/home/teleuser/Nellie/SPARC.toml"
group_name = "sparc"
run_duration = 300
chip = "sparc0"

# Create controller
ctrl = ScriptableController(group_name)

# Load configuration
cfg = load_config(config_file_path)

# Override output directory
cfg._satellite_configs["MessagePackWriter.SPARC".lower()]._dictionary["output_directory"] = f"/home/teleuser/Nellie/data/fhr/{chip}"

# Wait until all satellites are connected
ctrl.await_satellites(["MessagePackWriter.SPARC", "Keithley.Bias", "Caribou.SPARC"])

# Initialize
ctrl.constellation.initialize(cfg)
ctrl.await_state(SatelliteState.INIT, timeout=5)

# Launch
ctrl.constellation.launch()
ctrl.await_state(SatelliteState.ORBIT, timeout=30)


voltage_vcasb = [
    (0.0, range(10, 170, 10)),
    (-1.2, range(200, 360, 10)),
    (-3.0, range(430, 590, 10)),
]

for voltage, vcasb_range in voltage_vcasb:
    ctrl.log.status(f"Reconfiguring bias voltage to {voltage}V")
    recfg_keithley = {"voltage": voltage}
    ctrl.constellation.Keithley.Bias.reconfigure(recfg_keithley)
    ctrl.await_state(SatelliteState.ORBIT)

    for vcasb in vcasb_range:
        ctrl.log.status(f"Reconfuring VCASB to {vcasb}mV")

        # Reconfigure
        recfg = {"devices": {"SPARC": {"vcasb": vcasb * 1e-3}}}
        ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
        ctrl.await_state(SatelliteState.ORBIT)

        # Start the run
        run_identifier = f"run_{int(abs(voltage) * 10)}V_vcasb_{vcasb}"
        ctrl.log.status(f"Starting run {run_identifier}")
        ctrl.constellation.start(run_identifier)
        ctrl.await_state(SatelliteState.RUN)

        # Run for a while
        time.sleep(run_duration)

        # Stop the run and await ORBIT state of all satellites
        ctrl.constellation.stop()
        ctrl.await_state(SatelliteState.ORBIT)


# Land
ctrl.constellation.land()
ctrl.await_state(SatelliteState.INIT)
