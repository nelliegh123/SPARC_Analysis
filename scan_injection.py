from sparc_common import NCOLS, NROWS, ChipParams
from sparc_scanner import InjectionSettings, Scanner

# Settings
CONFIG_FILE = "SPARC.toml"
GROUP_NAME = "sparc"
DATA_DIR = "/home/teleuser/Nellie/data"
CHIP = "sparc7"
N_INJECTIONS = 5000
PULSE_INTERVAL = 100e-6  # 100us
COLS = range(NCOLS)
ROWS = range(NROWS)

scanner = Scanner(CONFIG_FILE, GROUP_NAME, DATA_DIR, CHIP)
scanner.setup("injection")
injs = InjectionSettings(N_INJECTIONS, PULSE_INTERVAL, COLS, ROWS)

# # DP optimized values
# for vh in list(range(30, 150, 10)) + list(range(200, 1300, 100)):
#     scanner.scan_iteration_inj(ChipParams(360, 70, 400, 700, 300, vh), injs)
# # DPTS nominal values
# for vh in list(range(30, 150, 10)) + list(range(200, 1300, 100)):
#     scanner.scan_iteration_inj(ChipParams(300, 10, 300, 100, 100, vh), injs)

# # Sparc6 iron values
for vh in list(range(30, 150, 10)) + list(range(200, 1300, 100)):
    scanner.scan_iteration_inj(ChipParams(360, 70, 400, 700, 300, vh), injs)
    #vcasb, ireset, vcasn, ibias, idb, vh


# Land
scanner.land()
