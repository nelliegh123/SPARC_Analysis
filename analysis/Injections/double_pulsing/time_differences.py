"""Reads the CSV of one run, splits the words into injections and plots t2 - t1, t3 - t2, ... as histograms."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
CSV_FILE = ROOT / "data" / "sparc7" / "double_pulsing" / "csv" / "pd_750.csv"
PLOT_DIR = ROOT / "plots" / "sparc7" / "double_pulsing" / "pd_750"


N_INJECTIONS = 1000  
GAP = 12_500_000     # ticks = 50 ms: a new injection starts when the FPGA timestamp jumps more than this
NS_PER_TICK = 4
N_HISTOGRAMS = 5     # t2 - t1, t3 - t2, ..., t6 - t5


fpga = []
combined = []
with open(CSV_FILE) as f:
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
print(f"Found {len(injections)} injections of {N_INJECTIONS}")
print(f"Found {len(combined)} entries")
print("=================================")
print()
print("---------------------------------")
for n in sorted(set(len(injection) for injection in injections)):
    count = sum(1 for injection in injections if len(injection) == n)
    print(f"  {count} injections with {n} threshold crossings")
print("---------------------------------")
print()


#------------Plotting many histograms, t2-t1, t3-t2, etc.----------------
PLOT_DIR.mkdir(parents=True, exist_ok=True)
for k in range(1, N_HISTOGRAMS + 1):
    differences = []
    for injection in injections:
        t = sorted(injection)
        if len(t) > k:
            differences.append((t[k] - t[k - 1]) * NS_PER_TICK)

    plt.figure()
    plt.hist(differences, bins=50)
    plt.xlabel(f"t{k + 1} - t{k} [ns]")
    plt.ylabel("Counts")
    plt.title(f"t{k + 1} - t{k}, {len(differences)} injections")
    output_file = PLOT_DIR / f"t{k + 1}_minus_t{k}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
print(f"Saved {output_file}")
print()




#--------Plotting only the injections with exactly two threshold crossings: t2 - t1-------
two_crossings = []

for injection in injections:
    t = sorted(injection)
    if len(t) == 2:
        two_crossings.append((t[1] - t[0]) * NS_PER_TICK)
# print(f"Found {len(two_crossings)} injections with two threshold crossings: t2 - t1 = {two_crossings} ns")

plt.figure()
plt.hist(two_crossings, bins=50)
plt.xlabel("t2 - t1 [ns]")
plt.ylabel("Counts")
plt.title(f"t2 - t1, only the {len(two_crossings)} injections with two threshold crossings")
output_file = PLOT_DIR / "two_crossings_t2_minus_t1.png"
plt.savefig(output_file, dpi=150)
plt.close()
print(f"Saved {output_file}")
print()




#------------Injections with exactly three threshold crossings: all their times----------------
three = []
for injection in injections:
    t = sorted(injection)
    if len(t) == 3:
        three.append([(x - t[0]) * NS_PER_TICK for x in t])  # times since t1 [ns]

# print("=============================================================")
# print(f"Found {len(three)} injections with three threshold crossings")
# print("=============================================================")
# print("  t1, t2, t3 since t1 [ns]")
# for s in three:
#     print(f"  {s}")
# print()

# stacked=True puts the histograms on top of each other, so the bin heights add up
plt.figure()
plt.hist([[s[1] - s[0] for s in three], [s[2] - s[1] for s in three]],
         bins=range(0, 8500, 100), stacked=True,
         color=["tab:blue", "tab:orange"], label=["t2 - t1", "t3 - t2"])
plt.xlabel("Time difference [ns]")
plt.ylabel("Counts")
plt.title(f"The {len(three)} injections with three threshold crossings")
plt.legend()
output_file = PLOT_DIR / "three_crossings.png"
plt.savefig(output_file, dpi=150)
plt.close()
print(f"Saved {output_file}")




#------------Injections with exactly four threshold crossings----------------
four = []
for injection in injections:
    t = sorted(injection)
    if len(t) == 4:
        four.append([(x - t[0]) * NS_PER_TICK for x in t])  # times since t1 [ns]

# stacked=True puts the histograms on top of each other, so the bin heights add up
plt.figure()
plt.hist([[s[1] - s[0] for s in four], [s[2] - s[1] for s in four], [s[3] - s[2] for s in four]],
         bins=range(0, 8500, 100), stacked=True,
         color=["tab:blue", "tab:orange", "tab:green"], label=["t2 - t1", "t3 - t2", "t4 - t3"])
plt.xlabel("Time difference [ns]")
plt.ylabel("Counts")
plt.title(f"The {len(four)} injections with four threshold crossings")
plt.legend()
output_file = PLOT_DIR / "four_crossings.png"
plt.savefig(output_file, dpi=150)
plt.close()
print(f"Saved {output_file}")
print()



#------------Injections with exactly five threshold crossings----------------
five = []
for injection in injections:
    t = sorted(injection)
    if len(t) == 5:
        five.append([(x - t[0]) * NS_PER_TICK for x in t])  # times since t1 [ns]

# print("=============================================================")
# print(f"Found {len(five)} injections with five threshold crossings")
# print("=============================================================")
# print("  t1, t2, t3, t4, t5 since t1 [ns]")
# for s in five:
#     print(f"  {s}")
# print()



plt.figure()
plt.hist([[s[1] - s[0] for s in five], [s[2] - s[1] for s in five], [s[3] - s[2] for s in five], [s[4] - s[3] for s in five]],
         bins=range(0, 8500, 100), stacked=True,
         color=["tab:blue", "tab:orange", "tab:green", "tab:red"], label=["t2 - t1", "t3 - t2", "t4 - t3", "t5 - t4"])
plt.xlabel("Time difference [ns]")
plt.ylabel("Counts")
plt.title(f"The {len(five)} injections with five threshold crossings")
plt.legend()
output_file = PLOT_DIR / "five_crossings.png"
plt.savefig(output_file, dpi=150)
plt.close()
print(f"Saved {output_file}")
print()






# #------------Closer look at the injections with t2 - t1 < 2000 ns----------------
# MAX_T2_T1 = 2000  # ns
# N_PRINT = 20      # injections printed crossing by crossing

# # Each short injection is a list of crossing times [ns] since its first crossing, so the first entry is always 0
# short = []
# for injection in injections:
#     t = sorted(injection)
#     if len(t) >= 2 and (t[1] - t[0]) * NS_PER_TICK < MAX_T2_T1:
#         short.append([(x - t[0]) * NS_PER_TICK for x in t])

# print("=============================================================")
# print(f"Found {len(short)} injections with t2 - t1 < {MAX_T2_T1} ns")
# print("=============================================================")
# for n in sorted(set(len(s) for s in short)):
#     count = sum(1 for s in short if len(s) == n)
#     print(f"  {count} of them with {n} threshold crossings")
# print()
# print(f"The first {N_PRINT}, crossing times since t1 [ns]:")
# for s in short[:N_PRINT]:
#     print(f"  {s}")
# print()

# # t2 - t1 of these injections, one bin per 4 ns tick
# plt.figure()
# plt.hist([s[1] for s in short], bins=range(0, MAX_T2_T1 + 4, 4))
# plt.xlabel("t2 - t1 [ns]")
# plt.ylabel("Injections per 4 ns")
# plt.title(f"t2 - t1 of the {len(short)} injections with t2 - t1 < {MAX_T2_T1} ns")
# output_file = PLOT_DIR / "short_t2_minus_t1.png"
# plt.savefig(output_file, dpi=150)
# plt.close()
# print(f"Saved {output_file}")

# # All crossings of these injections, as time since t1: shows where t3, t4, ... come
# plt.figure()
# plt.hist([x for s in short for x in s[1:]], bins=100)
# plt.xlabel("Time since t1 [ns]")
# plt.ylabel("Crossings")
# plt.title(f"t2, t3, ... of the {len(short)} injections with t2 - t1 < {MAX_T2_T1} ns")
# output_file = PLOT_DIR / "short_all_crossings.png"
# plt.savefig(output_file, dpi=150)
# plt.close()
# print(f"Saved {output_file}")
# print()
