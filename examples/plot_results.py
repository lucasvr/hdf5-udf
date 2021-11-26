#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys

no_sandbox_chunked_1000_cpp = [0.41, 0.34, 0.45]
no_sandbox_chunked_1000_cuda_1 = [2.64, 2.78, 2.69]
no_sandbox_chunked_1000_cuda_16 = [4.47, 4.63, 5.42]
no_sandbox_chunked_1000_cuda_2 = [2.69, 2.65, 2.62]
no_sandbox_chunked_1000_cuda_4 = [2.75, 2.85, 3.49]
no_sandbox_chunked_1000_cuda_8 = [5.0, 3.49, 5.62]
no_sandbox_chunked_1000_ds1 = [0.46, 0.51, 0.36]
no_sandbox_chunked_1000_lua = [0.37, 0.37, 0.36]
no_sandbox_chunked_1000_py = [1.44, 1.52, 1.49]

no_sandbox_contiguous_1000_cpp = [0.16, 0.03, 0.03]
no_sandbox_contiguous_1000_cuda_1 = [2.51, 2.56, 2.55]
no_sandbox_contiguous_1000_cuda_16 = [2.65, 2.66, 2.65]
no_sandbox_contiguous_1000_cuda_2 = [2.61, 2.62, 2.54]
no_sandbox_contiguous_1000_cuda_4 = [2.61, 2.65, 2.65]
no_sandbox_contiguous_1000_cuda_8 = [2.61, 2.63, 2.68]
no_sandbox_contiguous_1000_ds1 = [0.01, 0.01, 0.01]
no_sandbox_contiguous_1000_lua = [0.03, 0.03, 0.03]
no_sandbox_contiguous_1000_py = [1.11, 1.11, 1.08]

no_sandbox_chunked_16000_cpp = [7.59, 7.8, 7.75]
no_sandbox_chunked_16000_cuda_1 = [6.31, 5.52, 6.59]
no_sandbox_chunked_16000_cuda_16 = [3.97, 5.1, 5.01]
no_sandbox_chunked_16000_cuda_2 = [5.39, 6.18, 4.83]
no_sandbox_chunked_16000_cuda_4 = [7.09, 6.28, 6.69]
no_sandbox_chunked_16000_cuda_8 = [7.58, 5.71, 6.5]
no_sandbox_chunked_16000_ds1 = [3.05, 3.36, 3.52]
no_sandbox_chunked_16000_lua = [7.14, 7.13, 7.53]
no_sandbox_chunked_16000_py = [278.13, 280.0, 273.51]

no_sandbox_contiguous_16000_cpp = [4.81, 4.69, 4.62]
no_sandbox_contiguous_16000_cuda_1 = [4.44, 4.25, 4.33]
no_sandbox_contiguous_16000_cuda_16 = [3.5, 4.03, 3.97]
no_sandbox_contiguous_16000_cuda_2 = [3.75, 3.86, 3.87]
no_sandbox_contiguous_16000_cuda_4 = [3.76, 3.72, 3.62]
no_sandbox_contiguous_16000_cuda_8 = [3.58, 3.68, 3.56]
no_sandbox_contiguous_16000_ds1 = [3.13, 2.92, 2.92]
no_sandbox_contiguous_16000_lua = [3.57, 3.58, 3.6]
no_sandbox_contiguous_16000_py = [272.21, 277.8, 270.99]

no_sandbox_chunked_2000_cpp = [0.6, 0.57, 0.57]
no_sandbox_chunked_2000_cuda_1 = [2.93, 4.27, 4.59]
no_sandbox_chunked_2000_cuda_16 = [3.41, 3.41, 3.19]
no_sandbox_chunked_2000_cuda_2 = [4.41, 4.73, 4.54]
no_sandbox_chunked_2000_cuda_4 = [4.5, 4.14, 5.52]
no_sandbox_chunked_2000_cuda_8 = [3.7, 3.27, 3.07]
no_sandbox_chunked_2000_ds1 = [0.39, 0.75, 0.67]
no_sandbox_chunked_2000_lua = [0.54, 0.73, 0.5]
no_sandbox_chunked_2000_py = [4.8, 5.02, 4.79]

no_sandbox_contiguous_2000_cpp = [0.1, 0.09, 0.09]
no_sandbox_contiguous_2000_cuda_1 = [2.56, 2.59, 2.68]
no_sandbox_contiguous_2000_cuda_16 = [3.09, 3.36, 3.19]
no_sandbox_contiguous_2000_cuda_2 = [2.56, 2.61, 2.68]
no_sandbox_contiguous_2000_cuda_4 = [2.71, 2.61, 2.71]
no_sandbox_contiguous_2000_cuda_8 = [2.74, 2.88, 2.97]
no_sandbox_contiguous_2000_ds1 = [0.05, 0.05, 0.05]
no_sandbox_contiguous_2000_lua = [0.07, 0.07, 0.07]
no_sandbox_contiguous_2000_py = [4.15, 4.18, 4.23]

no_sandbox_chunked_32000_cpp = [27.66, 29.61, 29.48]
no_sandbox_chunked_32000_cuda_1 = [11.07, 12.77, 12.53]
no_sandbox_chunked_32000_cuda_16 = [10.53, 10.7, 10.64]
no_sandbox_chunked_32000_cuda_2 = [9.82, 9.68, 9.54]
no_sandbox_chunked_32000_cuda_4 = [9.81, 9.85, 9.89]
no_sandbox_chunked_32000_cuda_8 = [9.21, 10.2, 10.08]
no_sandbox_chunked_32000_ds1 = [10.58, 10.28, 9.21]
no_sandbox_chunked_32000_lua = [1356.66, 1348.67, 1358.82]
no_sandbox_chunked_32000_py = [1110.06, 1108.84, 1119.87]

no_sandbox_contiguous_32000_cpp = [18.19, 18.58, 18.55]
no_sandbox_contiguous_32000_cuda_1 = [10.02, 10.06, 10.05]
no_sandbox_contiguous_32000_cuda_16 = [7.87, 3.73, 8.39]
no_sandbox_contiguous_32000_cuda_2 = [8.69, 7.94, 7.99]
no_sandbox_contiguous_32000_cuda_4 = [7.06, 6.98, 7.0]
no_sandbox_contiguous_32000_cuda_8 = [7.99, 3.52, 5.65]
no_sandbox_contiguous_32000_ds1 = [12.29, 11.59, 11.62]
no_sandbox_contiguous_32000_lua = [14.62, 14.64, 13.95]
no_sandbox_contiguous_32000_py = [1107.24, 1102.42, 1110.65]

no_sandbox_chunked_4000_cpp = [1.28, 1.22, 1.56]
no_sandbox_chunked_4000_cuda_1 = [3.04, 3.16, 2.94]
no_sandbox_chunked_4000_cuda_16 = [6.27, 4.95, 6.78]
no_sandbox_chunked_4000_cuda_2 = [2.83, 3.19, 4.42]
no_sandbox_chunked_4000_cuda_4 = [4.25, 4.15, 3.36]
no_sandbox_chunked_4000_cuda_8 = [5.59, 3.43, 4.36]
no_sandbox_chunked_4000_ds1 = [1.01, 1.1, 0.94]
no_sandbox_chunked_4000_lua = [1.17, 1.34, 1.41]
no_sandbox_chunked_4000_py = [17.52, 18.11, 17.83]

no_sandbox_contiguous_4000_cpp = [0.31, 0.33, 0.33]
no_sandbox_contiguous_4000_cuda_1 = [2.75, 2.62, 2.67]
no_sandbox_contiguous_4000_cuda_16 = [3.0, 3.12, 3.22]
no_sandbox_contiguous_4000_cuda_2 = [2.64, 2.6, 2.99]
no_sandbox_contiguous_4000_cuda_4 = [2.81, 2.67, 2.77]
no_sandbox_contiguous_4000_cuda_8 = [2.92, 3.03, 4.12]
no_sandbox_contiguous_4000_ds1 = [0.2, 0.19, 0.19]
no_sandbox_contiguous_4000_lua = [0.25, 0.25, 0.25]
no_sandbox_contiguous_4000_py = [17.4, 17.46, 16.85]

no_sandbox_chunked_8000_cpp = [2.37, 3.25, 2.84]
no_sandbox_chunked_8000_cuda_1 = [4.35, 4.61, 5.53]
no_sandbox_chunked_8000_cuda_16 = [3.78, 3.58, 3.86]
no_sandbox_chunked_8000_cuda_2 = [5.1, 4.42, 4.89]
no_sandbox_chunked_8000_cuda_4 = [3.85, 3.24, 3.34]
no_sandbox_chunked_8000_cuda_8 = [3.38, 3.4, 3.43]
no_sandbox_chunked_8000_ds1 = [1.06, 1.06, 1.11]
no_sandbox_chunked_8000_lua = [2.28, 2.34, 2.32]
no_sandbox_chunked_8000_py = [69.49, 70.03, 70.44]

no_sandbox_contiguous_8000_cpp = [1.17, 1.2, 1.2]
no_sandbox_contiguous_8000_cuda_1 = [3.0, 2.97, 3.01]
no_sandbox_contiguous_8000_cuda_16 = [3.47, 3.16, 3.36]
no_sandbox_contiguous_8000_cuda_2 = [2.94, 2.86, 3.03]
no_sandbox_contiguous_8000_cuda_4 = [2.86, 3.1, 2.98]
no_sandbox_contiguous_8000_cuda_8 = [3.13, 3.08, 3.09]
no_sandbox_contiguous_8000_ds1 = [0.79, 0.76, 0.76]
no_sandbox_contiguous_8000_lua = [0.92, 0.92, 0.92]
no_sandbox_contiguous_8000_py = [68.92, 68.65, 68.35]


myself = sys.modules[__name__]
modes = ["contiguous", "chunked"]
### grid_sizes = ['1000', '16000', '2000', '32000', '4000', '8000']
grid_sizes = ["1000", "2000", "4000", "8000", "16000", "32000"]

def compare_lang(entry):
  # langs = ['cpp', 'lua', 'py', 'cuda_1', 'cuda_16', 'cuda_2', 'cuda_32', 'cuda_4', 'cuda_8']
  return int(entry.split("_")[1]) if entry.find("_") > 0 else ord(entry[0])*100 + ord(entry[1])

sorted_langs = ['cpp', 'cuda_1', 'cuda_16', 'cuda_2', 'cuda_4', 'cuda_8', 'ds1', 'lua', 'py']
sorted_langs.sort(key=compare_lang)
sorted_langs = ['cuda_1', 'cuda_2', 'cuda_4', 'cuda_8', 'cuda_16', 'cpp', 'lua']
### sorted_langs = ['cpp', 'lua', 'py', 'ds1'] # XXX: sandbox vs no-sandbox comparison


fmt_styles = []
pretty_names = []
for lang in sorted_langs:
    lang = lang.replace('cpp', 'C++').replace('lua', 'LuaJIT').replace('py', 'CPython').replace('ds1', 'Reference')
    ### lang = lang.replace('cpp', 'C++ UDF').replace('lua', 'LuaJIT UDF').replace('py', 'CPython UDF').replace('ds1', 'Contiguous dataset')
    if lang.startswith('cuda'):
        n = int(lang.split('_')[1])
        stream = 'stream' if n == 1 else 'streams'
        lang = 'CUDA ({} {})'.format(n, stream)
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
        plt.errorbar(grid_sizes, datapoints_avg, yerr=datapoints_std, fmt=fmt_styles[j], label=label)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.title("NDVI {} dataset read times".format(mode), fontname='Times New Roman', fontsize=12)
    plt.xlabel("Dataset size", fontname='Times New Roman', fontsize=10)
    plt.ylabel("Time (secs)", fontname='Times New Roman', fontsize=10)
    plt.legend(loc="upper left")
    plt.yscale("log")
    plt.savefig("perf-no_sandbox-{}-logscale.pdf".format(mode))
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
    
        ax.set_title("UDF overhead ({}x{})".format(grid_size, grid_size), fontname='Times New Roman', fontsize=12)
        ax.set_xticks(ind)
        ax.set_xticklabels([x for x in pretty_names if not x.startswith('CUDA')])
        ax.set_ylabel("Time (secs)", fontname='Times New Roman', fontsize=10)
        ax.label_outer()
        if i == 0:
            ax.legend(bbox_to_anchor=(1.08, 0.98), fancybox=True, shadow=True, loc='upper right')
        
    fig.tight_layout()
    plt.savefig("perf-sandbox_overhead-{}-logscale.pdf".format(mode))



