#!/usr/bin/env python3

import os
import glob

contents = """#!/usr/bin/env python3\n
import matplotlib.pyplot as plt
import numpy as np
import sys\n
"""

# Get running times from results/numbers.${lang}
def get_numbers_lowres(path):
    # 0:00.16
    times = [x.split(" ")[-1] for x in open(path).readlines() if x.find("wall clock") > 0]
    times_in_sec = [int(x.split(":")[0]) * 60 + float(x.split(":")[1]) for x in times]
    return times_in_sec

# Get running times from results/internal_numbers.${lang}
def get_numbers_highres(path):
    times_in_sec = []
    with open(path) as f:
        for line in f.readlines():
            if line.startswith('Call to'):
                times_in_sec += [ds1 + ds2]
                ds1, ds2 = None, None
            elif line.find('Dataset1') >= 0:
                ds1 = float(line.split()[-2])
            elif line.find('Dataset2') >= 0:
                ds2 = float(line.split()[-2])
    return times_in_sec

grid_sizes = []
langs = []

for name in sorted(glob.glob("results-sandbox/results*")) + sorted(glob.glob("results-no_sandbox/results*")):
    size = os.path.basename(name).split("-")[1].split("x")[0]  # 1000, 5000, etc
    mode = os.path.basename(name).split("-")[-1] # chunked, contiguous
    sandbox = "sandbox" if name.find("no_sandbox") < 0 else "no_sandbox"
    if not f"{size}" in grid_sizes:
        grid_sizes.append(f"{size}")

    for result in sorted(glob.glob(name + "/internal_numbers.*") + glob.glob(name + "/noop.*") + glob.glob(name + "/one_dep.*")):
        lang = os.path.basename(result).split(".")[1].replace("_parallel", "") # cpp, lua, py, cuda
        threads = os.path.basename(result).split(".")[2] if lang == "cuda" else ""
        test_name = f"{sandbox}_{mode}_{size}_{lang}"
 
        if len(threads):
            test_name += f"_{threads}"
            lang += f"_{threads}"

        if result.find("noop") > 0:
            test_name += "_noop"

        if result.find("one_dep") > 0:
            test_name += "_one_dep"

        if not lang in langs:
            langs.append(lang)

        numbers = get_numbers_highres(result)
        data = f"{test_name} = {str(numbers)}\n"
        contents += data

    contents += "\n"


contents += f"""
myself = sys.modules[__name__]
modes = ["contiguous", "chunked"]
### grid_sizes = {grid_sizes}
grid_sizes = ["1000", "2000", "4000", "8000", "16000", "32000"]

def compare_lang(entry):
  # langs = ['cpp', 'lua', 'py', 'cuda_1', 'cuda_16', 'cuda_2', 'cuda_32', 'cuda_4', 'cuda_8']
  return int(entry.split("_")[1]) if entry.find("_") > 0 else ord(entry[0])*100 + ord(entry[1])

sorted_langs = {langs}
sorted_langs.sort(key=compare_lang)
sorted_langs = ['cuda_1', 'cuda_2', 'cuda_4', 'cuda_8', 'cuda_16', 'cpp']
### sorted_langs = ['cpp', 'lua', 'py', 'ds1'] # XXX: sandbox vs no-sandbox comparison


fmt_styles = []
pretty_names = []
for lang in sorted_langs:
    lang = lang.replace('cpp', 'C++').replace('lua', 'LuaJIT').replace('py', 'CPython').replace('ds1', 'Reference')
    ### lang = lang.replace('cpp', 'C++ UDF').replace('lua', 'LuaJIT UDF').replace('py', 'CPython UDF').replace('ds1', 'Contiguous dataset')
    if lang.startswith('cuda'):
        n = int(lang.split('_')[1])
        stream = 'stream' if n == 1 else 'streams'
        lang = 'CUDA ({{}} {{}})'.format(n, stream)
    pretty_names.append(lang)
    fmt_styles.append('-o' if lang.startswith('CUDA') else '--o')

print("No-sandbox (GDS)")

# No-sandbox
for i, mode in enumerate(modes):
    for j, lang in enumerate(sorted_langs):
        label = pretty_names[j]
        if mode == 'contiguous' and label.startswith('CUDA'):
            label = label.replace('streams', 'threads')
        datapoints_avg = [np.average(getattr(myself, "no_sandbox_" + mode + "_" + grid + "_" + lang)) for grid in grid_sizes]
        datapoints_std = [np.std(getattr(myself, "no_sandbox_" + mode + "_" + grid + "_" + lang)) for grid in grid_sizes]
        plt.plot(grid_sizes, datapoints_avg, fmt_styles[j], label=label)
        #plt.errorbar(grid_sizes, datapoints_avg, yerr=datapoints_std, fmt=fmt_styles[j], label=label)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.title("NDVI {{}} dataset read times".format(mode), fontname='Times New Roman', fontsize=12)
    plt.xlabel("Dataset size", fontname='Times New Roman', fontsize=10)
    plt.ylabel("Time (secs)", fontname='Times New Roman', fontsize=10)
    plt.legend(loc="upper left")
    plt.yscale("log")
    plt.savefig("perf-no_sandbox-{{}}-logscale.pdf".format(mode))
    plt.cla()
    plt.clf()
    plt.close()

sys.exit(0)

print("Sandbox x no-sandbox comparison")        

# Sandbox and no-sandbox comparison. CUDA is left out, as we don't have a sandbox
# implementation of that backend yet.
modes = ["contiguous"] # xxx
grid_sizes = ["1000", "16000"] # xxx
for i, mode in enumerate(modes):
    ind = np.arange(len([x for x in sorted_langs if not x.startswith('cuda')]))
    bar_width = 0.15
    
    fig, axs = plt.subplots(2)

    # We only want to check the overhead of sandbox with the smallest and biggest grid sizes
    for i, grid_size in enumerate([grid_sizes[0], grid_sizes[-1]]):
        sandbox_avgs, sandbox_stds = [], []
        no_sandbox_avgs, no_sandbox_stds = [], []
        noop_avgs, noop_stds = [], []
        one_dep_avgs, one_dep_stds = [], []

        for j, lang in enumerate(sorted_langs):
            if not lang.startswith('cuda'):
                #sandbox_avgs.append(np.average(getattr(myself, "sandbox_" + mode + "_" + grid_size + "_" + lang)))
                #sandbox_stds.append(np.std(getattr(myself, "sandbox_" + mode + "_" + grid_size + "_" + lang)))
                if lang == "ds1":
                    noop_avgs.append(0)
                    noop_stds.append(0)
                    one_dep_avgs.append(0)
                    one_dep_stds.append(0)
                    no_sandbox_avgs.append(np.average(getattr(myself, "no_sandbox_" + mode + "_" + grid_size + "_" + lang)))
                    no_sandbox_stds.append(np.std(getattr(myself, "no_sandbox_" + mode + "_" + grid_size + "_" + lang)))
                else:
                    noop_avgs.append(np.average(getattr(myself, "no_sandbox_" + mode + "_" + grid_size + "_" + lang + "_noop")))
                    noop_stds.append(np.std(getattr(myself, "no_sandbox_" + mode + "_" + grid_size + "_" + lang + "_noop")))
                    one_dep_avgs.append(np.average(getattr(myself, "no_sandbox_" + mode + "_" + grid_size + "_" + lang + "_one_dep")))
                    one_dep_stds.append(np.std(getattr(myself, "no_sandbox_" + mode + "_" + grid_size + "_" + lang + "_one_dep")))
                    no_sandbox_avgs.append(0)
                    no_sandbox_stds.append(0)

        ax = axs[i]
        ax.bar(ind-bar_width/2, noop_avgs, bar_width, yerr=noop_stds, label='No-op UDF, no deps')
        ax.bar(ind+bar_width/2, one_dep_avgs, bar_width, yerr=one_dep_stds, label='No-op UDF, 1 dep')
        ax.bar(ind, no_sandbox_avgs, bar_width, yerr=no_sandbox_stds, label='Dependency')
    
        ax.set_title("UDF overhead ({{}}x{{}})".format(grid_size, grid_size), fontname='Times New Roman', fontsize=12)
        ax.set_xticks(ind)
        ax.set_xticklabels([x for x in pretty_names if not x.startswith('CUDA')])
        ax.set_ylabel("Time (secs)", fontname='Times New Roman', fontsize=10)
        ax.label_outer()
        if i == 0:
            ax.legend(bbox_to_anchor=(1.08, 0.98), fancybox=True, shadow=True, loc='upper right')
        
    fig.tight_layout()
    plt.savefig("perf-sandbox_overhead-{{}}-logscale.pdf".format(mode))



"""

with open("plot_results.py", "w") as f:
    f.write(contents)
