import time

from constellation.core.controller import ScriptableController
from constellation.core.controller_configuration import load_config
from constellation.core.protocol.cscp1 import SatelliteState

import sparc_masking_generator

# Settings
config_file_path = "/home/teleuser/Nellie/SPARC.toml"
group_name = "sparc"
n_injections = 25
pulse_interval = 100e-6
chip = "sparc7"

# Create controller
ctrl = ScriptableController(group_name)

# Load configuration
cfg = load_config(config_file_path)

# Wait until all satellites are connected
ctrl.await_satellites(["MessagePackWriter.SPARC", "Keithley.Bias", "Caribou.SPARC"])

# Initialize
ctrl.constellation.initialize(cfg)
ctrl.await_state(SatelliteState.INIT, timeout=5)

# Launch
ctrl.constellation.launch()
ctrl.await_state(SatelliteState.ORBIT, timeout=30)

# Reconfigure output directory
ctrl.constellation.MessagePackWriter.SPARC.reconfigure({"output_directory": f"/home/teleuser/Nellie/data/{chip}/s_curve"})
ctrl.await_state(SatelliteState.ORBIT, timeout=5)

def scan_iteration(vcasb: float, vh: float) -> None:
    ctrl.log.status(f"Reconfiguring vcasb to {vcasb} mV and vh to {vh} mV")

    # Reconfigure
    recfg = {"devices": {"SPARC": {"vcasb": vcasb*1e-3, "vh": vh*1e-3}}}
    ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
    ctrl.await_state(SatelliteState.ORBIT)

    for col in range(sparc_masking_generator.NCOLS):
        for row in range(sparc_masking_generator.NROWS):
            ctrl.log.status(f"Reconfiguring active pixel to ({col}, {row})")

            # Reconfigure
            recfg = {"devices": {"SPARC": {"registers": sparc_masking_generator.sparc_mask_all_pixels_except(col, row)}}}
            ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
            ctrl.await_state(SatelliteState.ORBIT)

            # Start the run
            run_identifier = f"run_vcasb_{vcasb}_vh_{vh}_px_{col}_{row}"
            ctrl.log.status(f"Starting run {run_identifier}")
            ctrl.constellation.start(run_identifier)
            ctrl.await_state(SatelliteState.RUN)

            # Run injections
            for _ in range(n_injections):
                ctrl.constellation.Caribou.SPARC._set_memory(["sparc", "pulse_cmd", 1])
                time.sleep(pulse_interval)

            # Stop the run and await ORBIT state of all satellites
            ctrl.constellation.stop()
            ctrl.await_state(SatelliteState.ORBIT)


#--------Used this for sparc7-----------
# # Scan vcasb 300
# for vh in list(range(25, 225, 25)) + list(range(300, 1300, 100)):
#     scan_iteration(300, vh)
#---------------------------------------

# First iteration: fine vcasb scan with nominal vh
#for vcasb in range(220, 380, 10):
#    if vcasb not in [250, 300]:
#        scan_iteration(vcasb, 600)

# Second iteration: fine vh scan with a few vcasb values
#for vcasb in [250, 300]:
#    for vh in list(range(25, 225, 25)) + list(range(300, 1300, 100)):
#        scan_iteration(vcasb, vh)

for vh in list(range(25, 130, 5)):
    scan_iteration(180, vh)


# Land
ctrl.constellation.land()
ctrl.await_state(SatelliteState.INIT)
