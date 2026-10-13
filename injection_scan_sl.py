import time

from constellation.core.controller import ScriptableController
from constellation.core.controller_configuration import load_config
from constellation.core.protocol.cscp1 import SatelliteState

import sparc_masking_generator

# Settings
config_file_path = "/home/teleuser/Nellie/SPARC.toml"
group_name = "sparc"
n_injections = 1000
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
# ctrl.constellation.MessagePackWriter.SPARC.reconfigure({"output_directory": f"/home/teleuser/Nellie/data/{chip}/s_curve_300"})
ctrl.constellation.MessagePackWriter.SPARC.reconfigure({"output_directory": f"/home/teleuser/Nellie/data/{chip}/s_curve_fine"})
ctrl.await_state(SatelliteState.ORBIT, timeout=5)

# cols: the columns to scan, all of them if not given
def scan_iteration(vcasb: float, vh: float, cols=None) -> None:
    ctrl.log.status(f"Reconfiguring vcasb to {vcasb} mV and vh to {vh} mV")

    # Reconfigure
    recfg = {"devices": {"SPARC": {"vcasb": vcasb*1e-3, "vh": vh*1e-3}}}
    ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
    ctrl.await_state(SatelliteState.ORBIT)

    for col in (range(sparc_masking_generator.NCOLS) if cols is None else cols):
        for row in range(sparc_masking_generator.NROWS):
            ctrl.log.status(f"Reconfiguring active pixel to ({col}, {row})")

            # Reconfigure
            recfg = {"devices": {"SPARC": {"registers": sparc_masking_generator.sparc_mask_all_pixels_except(col, row)}}}
            ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
            ctrl.await_state(SatelliteState.ORBIT)

            # Start the run
            # Constellation only allows [A-Za-z0-9_-] in run identifiers, so 57.5 mV is written as 57p5
            run_identifier = f"run_vcasb_{vcasb}_vh_{str(vh).replace('.', 'p')}_px_{col}_{row}"
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

#-------------S-curve-------------
# 45, 50 and 55 mV are already done (data/sparc7/s_curve_fine), so this resumes at 60 mV
for vh in list(range(60, 130, 5)):
    scan_iteration(300, vh)

# Later: fill in the 2.5 mV points over the transition region (57.5-112.5 mV)
# for vh in [x / 2 for x in range(115, 230, 10)]:
#     scan_iteration(300, vh)
#---------------------------------

# #-------------Rerun of Vh = 900 mV for sparc7: columns 14-18 had ToT about twice too long-------------
# for vh in [900]:
#     scan_iteration(300, vh)  # columns 14 to 18
#-----------------------------------------------------------------------------------------------------

# Land
ctrl.constellation.land()
ctrl.await_state(SatelliteState.INIT)
