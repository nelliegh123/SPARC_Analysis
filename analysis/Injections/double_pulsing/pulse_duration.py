"""Reads the CSVs of the pd 250, 500 and 750 runs, splits the words into injections and prints how many there are."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = ROOT / "data" / "sparc7" / "double_pulsing" / "csv"
PLOT_DIR = ROOT / "plots" / "sparc7" / "double_pulsing" / "pulse_duration"
PULSE_DURATIONS = [250, 500, 750]


N_INJECTIONS = 1000
GAP = 12_500_000     # ticks = 50 ms: a new injection starts when the FPGA timestamp jumps more than this
MAX_LISTED = 8       # injections with more entries than this are summed up in one line
NS_PER_TICK = 4


t2_minus_t1 = {}  
two_entries = {}  

for pd in PULSE_DURATIONS:
    fpga = []
    combined = []
    with open(CSV_DIR / f"pd_{pd}.csv") as f:
        for row in csv.DictReader(f):
            fpga.append(int(row["fpga_timestamp"]))
            combined.append(int(row["combined_timestamp"]))

    injections = [[combined[0]]]
    for i in range(1, len(fpga)):
        if fpga[i] - fpga[i - 1] > GAP:
            injections.append([])
        injections[-1].append(combined[i])

    print()
    print("=================================")
    print(f"pd {pd}")
    print(f"Found {len(injections)} injections of {N_INJECTIONS}")
    print(f"Found {len(combined)} entries")
    print("=================================")
    for n in range(2, MAX_LISTED + 1):
        count = sum(1 for injection in injections if len(injection) == n)
        print(f"  {count} injections with {n} entries")
    count = sum(1 for injection in injections if len(injection) > MAX_LISTED)
    print(f"  {count} injections with more than {MAX_LISTED} entries")

    t2_minus_t1[pd] = []
    for injection in injections:
        t = sorted(injection)
        if len(t) >= 2:
            t2_minus_t1[pd].append((t[1] - t[0]) * NS_PER_TICK)

    two_entries[pd] = []
    for injection in injections:
        t = sorted(injection)
        if len(t) == 2:
            two_entries[pd].append((t[1] - t[0]) * NS_PER_TICK)
print()


#------------Plotting t2 - t1 for the three pulse durations side by side----------------
PLOT_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, len(PULSE_DURATIONS), figsize=(18, 5))
for ax, pd in zip(axes, PULSE_DURATIONS):
    ax.hist(t2_minus_t1[pd], bins=50)
    ax.set_xlabel("t2 - t1 [ns]")
    ax.set_ylabel("Counts")
    ax.set_title(f"pd {pd}, {len(t2_minus_t1[pd])} injections")
fig.tight_layout()
output_file = PLOT_DIR / "t2_minus_t1.png"
fig.savefig(output_file, dpi=150)
plt.close(fig)
print(f"Saved {output_file}")
print()

#-------------------plotting t2-t1 for the three pulse durations in one plot, for the injections with only two entries----
plt.figure()
plt.hist([two_entries[pd] for pd in PULSE_DURATIONS], bins=20,
         label=[f"pd {pd}, {len(two_entries[pd])} injections" for pd in PULSE_DURATIONS])
plt.xlabel("t2 - t1 [ns]")
plt.ylabel("Counts")
plt.gca().yaxis.get_major_locator().set_params(integer=True)  # only whole counts on the y axis
plt.title("t2 - t1, only the injections with two entries")
plt.legend()
output_file = PLOT_DIR / "two_entries_t2_minus_t1.png"
plt.savefig(output_file, dpi=150)
plt.close()
print(f"Saved {output_file}")
print()

for pd in PULSE_DURATIONS:
    print(f"pd {pd}: t2 - t1 = {two_entries[pd]} ns")
print()