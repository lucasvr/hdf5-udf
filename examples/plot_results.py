#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys

sandbox_chunked_1000_cpp = [2.11, 2.11, 2.14, 2.08, 2.06]
sandbox_chunked_1000_ds1 = [1.7, 1.73, 1.69, 1.69, 1.68]
sandbox_chunked_1000_lua = [2.14, 2.05, 2.07, 2.11, 2.14]
sandbox_chunked_1000_py = [2.68, 2.62, 2.52, 2.52, 2.5]

sandbox_contiguous_1000_cpp = [0.35, 0.28, 0.25, 0.34, 0.27]
sandbox_contiguous_1000_ds1 = [0.14, 0.1, 0.1, 0.1, 0.11]
sandbox_contiguous_1000_lua = [0.31, 0.21, 0.31, 0.19, 0.22]
sandbox_contiguous_1000_py = [0.71, 0.7, 0.72, 0.66, 0.65]

sandbox_chunked_16000_cpp = [21.85, 21.5, 22.13, 21.91, 21.43]
sandbox_chunked_16000_ds1 = [9.75, 9.87, 9.92, 9.89, 9.83]
sandbox_chunked_16000_lua = [21.41, 21.3, 21.5, 21.53, 21.77]
sandbox_chunked_16000_py = [73.99, 73.76, 73.69, 73.43, 73.28]

sandbox_contiguous_16000_cpp = [12.63, 12.97, 13.96, 12.77, 13.78]
sandbox_contiguous_16000_ds1 = [4.96, 5.5, 5.38, 5.53, 5.87]
sandbox_contiguous_16000_lua = [12.52, 13.18, 13.25, 13.28, 13.37]
sandbox_contiguous_16000_py = [65.24, 67.12, 65.77, 65.97, 65.92]

sandbox_chunked_2000_cpp = [3.0, 2.91, 2.92, 2.95, 2.88]
sandbox_chunked_2000_ds1 = [2.05, 2.06, 2.07, 2.06, 2.09]
sandbox_chunked_2000_lua = [3.01, 2.88, 2.94, 2.92, 2.88]
sandbox_chunked_2000_py = [3.98, 3.95, 3.95, 4.03, 3.97]

sandbox_contiguous_2000_cpp = [0.46, 0.47, 0.48, 0.38, 0.5]
sandbox_contiguous_2000_ds1 = [0.17, 0.18, 0.2, 0.25, 0.15]
sandbox_contiguous_2000_lua = [0.51, 0.55, 0.46, 0.54, 0.49]
sandbox_contiguous_2000_py = [1.56, 1.6, 1.68, 1.59, 1.46]

sandbox_chunked_4000_cpp = [4.63, 4.79, 4.67, 4.63, 4.67]
sandbox_chunked_4000_ds1 = [2.95, 2.87, 2.9, 2.91, 2.95]
sandbox_chunked_4000_lua = [4.68, 4.67, 4.8, 4.75, 4.72]
sandbox_chunked_4000_py = [8.12, 8.2, 8.08, 8.31, 8.92]

sandbox_contiguous_4000_cpp = [1.05, 1.12, 1.03, 1.07, 1.36]
sandbox_contiguous_4000_ds1 = [0.47, 0.48, 0.46, 0.56, 0.55]
sandbox_contiguous_4000_lua = [0.88, 0.93, 1.08, 1.17, 1.15]
sandbox_contiguous_4000_py = [4.59, 4.57, 4.78, 4.6, 4.55]

sandbox_chunked_8000_cpp = [9.27, 9.34, 9.36, 9.27, 9.05]
sandbox_chunked_8000_ds1 = [4.96, 4.83, 4.76, 4.77, 4.89]
sandbox_chunked_8000_lua = [9.16, 9.47, 9.31, 9.18, 9.16]
sandbox_chunked_8000_py = [22.56, 22.47, 22.63, 22.15, 22.44]

sandbox_contiguous_8000_cpp = [3.39, 3.62, 3.63, 3.73, 3.32]
sandbox_contiguous_8000_ds1 = [1.53, 1.33, 1.36, 1.52, 1.3]
sandbox_contiguous_8000_lua = [3.48, 3.39, 3.23, 3.34, 3.49]
sandbox_contiguous_8000_py = [16.62, 16.78, 16.48, 16.47, 18.0]

no_sandbox_chunked_1000_cpp = [2.16, 2.08, 2.1, 2.13, 2.16]
no_sandbox_chunked_1000_cuda_1 = [2.39, 2.14, 2.16, 2.23, 2.13]
no_sandbox_chunked_1000_cuda_16 = [1.61, 1.64, 1.52, 1.58, 1.5]
no_sandbox_chunked_1000_cuda_2 = [1.97, 1.89, 1.82, 1.89, 1.82]
no_sandbox_chunked_1000_cuda_32 = [1.66, 1.56, 1.56, 1.56, 1.62]
no_sandbox_chunked_1000_cuda_4 = [1.68, 1.72, 1.78, 1.66, 1.64]
no_sandbox_chunked_1000_cuda_8 = [1.86, 1.66, 1.76, 1.6, 1.62]
no_sandbox_chunked_1000_ds1 = [1.74, 1.76, 1.76, 1.67, 1.76]
no_sandbox_chunked_1000_lua = [2.14, 2.11, 2.13, 2.14, 2.12]
no_sandbox_chunked_1000_py = [2.59, 2.56, 2.57, 2.55, 2.56]

no_sandbox_contiguous_1000_cpp_noop = [0.34, 0.19, 0.27, 0.21, 0.18, 0.32, 0.29, 0.18, 0.19, 0.21]
no_sandbox_contiguous_1000_lua_noop = [0.22, 0.17, 0.19, 0.2, 0.18, 0.22, 0.21, 0.17, 0.25, 0.19]
no_sandbox_contiguous_1000_py_noop = [0.5, 0.38, 0.37, 0.44, 0.36, 0.44, 0.38, 0.34, 0.41, 0.4]
no_sandbox_contiguous_1000_cpp = [0.23, 0.28, 0.25, 0.23, 0.36]
no_sandbox_contiguous_1000_cuda_1 = [1.64, 1.37, 1.46, 1.47, 1.39]
no_sandbox_contiguous_1000_cuda_16 = [1.51, 1.73, 1.57, 1.43, 1.45]
no_sandbox_contiguous_1000_cuda_2 = [1.46, 1.39, 1.41, 1.53, 1.41]
no_sandbox_contiguous_1000_cuda_32 = [1.49, 1.5, 1.51, 1.54, 1.63]
no_sandbox_contiguous_1000_cuda_4 = [1.56, 1.45, 1.46, 1.49, 1.54]
no_sandbox_contiguous_1000_cuda_8 = [1.53, 1.47, 1.46, 1.48, 1.51]
no_sandbox_contiguous_1000_ds1 = [0.11, 0.21, 0.11, 0.1, 0.09]
no_sandbox_contiguous_1000_lua = [0.36, 0.22, 0.21, 0.39, 0.26]
no_sandbox_contiguous_1000_py = [0.81, 0.66, 0.68, 0.69, 0.77]
no_sandbox_contiguous_1000_cpp_one_dep = [0.34, 0.21, 0.22, 0.24, 0.24, 0.19, 0.21, 0.31, 0.27, 0.21]
no_sandbox_contiguous_1000_lua_one_dep = [0.22, 0.18, 0.25, 0.19, 0.26, 0.19, 0.2, 0.24, 0.17, 0.22]
no_sandbox_contiguous_1000_py_one_dep = [0.37, 0.44, 0.38, 0.4, 0.46, 0.38, 0.36, 0.38, 0.38, 0.38]

no_sandbox_chunked_16000_cpp = [21.47, 22.35, 22.02, 21.18, 21.12]
no_sandbox_chunked_16000_cuda_1 = [16.24, 16.33, 16.31, 16.36, 16.36]
no_sandbox_chunked_16000_cuda_16 = [7.94, 8.55, 8.65, 13.38, 8.7]
no_sandbox_chunked_16000_cuda_2 = [15.46, 15.65, 15.64, 15.4, 15.78]
no_sandbox_chunked_16000_cuda_32 = [8.5, 8.2, 7.98, 8.57, 8.29]
no_sandbox_chunked_16000_cuda_4 = [10.16, 10.08, 10.26, 10.22, 10.07]
no_sandbox_chunked_16000_cuda_8 = [9.21, 9.21, 8.88, 9.65, 9.39]
no_sandbox_chunked_16000_ds1 = [9.58, 9.69, 9.56, 9.58, 9.73]
no_sandbox_chunked_16000_lua = [21.22, 21.1, 21.18, 21.6, 21.62]
no_sandbox_chunked_16000_py = [72.5, 72.6, 72.48, 72.73, 72.25]

no_sandbox_contiguous_16000_cpp_noop = [2.55, 2.42, 2.38, 2.45, 2.46, 2.76, 2.47, 2.47, 2.41, 2.44, 2.48, 2.51, 2.5, 2.47, 2.42]
no_sandbox_contiguous_16000_lua_noop = [2.53, 2.45, 2.39, 2.55, 2.45, 2.44, 2.46, 2.41, 2.44, 2.41, 2.4, 2.51, 2.44, 2.42, 2.42]
no_sandbox_contiguous_16000_py_noop = [2.73, 2.58, 2.61, 2.55, 2.65, 2.74, 2.63, 2.63, 2.58, 2.65, 2.62, 2.65, 2.71, 2.65, 2.63]
no_sandbox_contiguous_16000_cpp = [12.89, 13.27, 13.27, 11.94, 12.99]
no_sandbox_contiguous_16000_cuda_1 = [3.02, 3.02, 2.93, 2.97, 2.97]
no_sandbox_contiguous_16000_cuda_16 = [12.57, 13.11, 12.9, 12.99, 12.61]
no_sandbox_contiguous_16000_cuda_2 = [9.53, 7.43, 8.08, 8.43, 8.91]
no_sandbox_contiguous_16000_cuda_32 = [12.88, 12.57, 12.51, 12.87, 15.26]
no_sandbox_contiguous_16000_cuda_4 = [19.22, 19.03, 19.61, 18.5, 19.4]
no_sandbox_contiguous_16000_cuda_8 = [14.86, 14.27, 13.85, 14.08, 14.73]
no_sandbox_contiguous_16000_ds1 = [5.49, 5.27, 5.39, 5.38, 5.16]
no_sandbox_contiguous_16000_lua = [12.34, 12.67, 13.35, 13.31, 12.44]
no_sandbox_contiguous_16000_py = [74.0, 74.39, 74.12, 73.95, 74.64]
no_sandbox_contiguous_16000_cpp_one_dep = [7.43, 7.67, 8.16, 7.63, 7.79, 7.99, 7.56, 7.54, 7.67, 7.77, 7.92, 7.62, 8.19, 7.84]
no_sandbox_contiguous_16000_lua_one_dep = [7.57, 7.89, 7.61, 7.94, 7.71, 7.74, 7.24, 7.52, 7.63, 7.61, 8.24, 8.0, 7.72, 7.76, 7.81]
no_sandbox_contiguous_16000_py_one_dep = [8.06, 7.95, 8.01, 7.88, 7.94, 7.46, 7.63, 8.25, 7.84, 7.77, 7.94, 7.58, 7.86, 7.6, 8.67]

no_sandbox_chunked_2000_cpp = [2.97, 2.99, 2.96, 2.91, 3.01]
no_sandbox_chunked_2000_cuda_1 = [2.83, 2.85, 3.09, 2.92, 2.85]
no_sandbox_chunked_2000_cuda_16 = [1.69, 1.69, 1.68, 1.71, 1.76]
no_sandbox_chunked_2000_cuda_2 = [2.24, 2.27, 2.31, 2.28, 2.35]
no_sandbox_chunked_2000_cuda_32 = [1.67, 1.71, 1.81, 1.69, 1.72]
no_sandbox_chunked_2000_cuda_4 = [2.01, 1.94, 2.13, 1.93, 1.88]
no_sandbox_chunked_2000_cuda_8 = [1.76, 1.79, 1.78, 1.79, 1.8]
no_sandbox_chunked_2000_ds1 = [2.12, 2.17, 2.11, 2.23, 2.11]
no_sandbox_chunked_2000_lua = [2.89, 2.88, 2.92, 2.9, 2.94]
no_sandbox_chunked_2000_py = [4.04, 3.99, 4.01, 4.08, 4.01]

no_sandbox_contiguous_2000_cpp = [0.44, 0.44, 0.55, 0.48, 0.56]
no_sandbox_contiguous_2000_cuda_1 = [1.52, 1.54, 1.5, 1.47, 1.49]
no_sandbox_contiguous_2000_cuda_16 = [1.57, 1.77, 1.62, 1.6, 1.68]
no_sandbox_contiguous_2000_cuda_2 = [1.59, 1.63, 1.58, 1.51, 1.58]
no_sandbox_contiguous_2000_cuda_32 = [1.65, 1.65, 1.73, 1.73, 1.76]
no_sandbox_contiguous_2000_cuda_4 = [1.66, 1.8, 1.72, 1.66, 1.65]
no_sandbox_contiguous_2000_cuda_8 = [1.62, 1.61, 1.7, 1.75, 1.67]
no_sandbox_contiguous_2000_ds1 = [0.18, 0.32, 0.19, 0.23, 0.16]
no_sandbox_contiguous_2000_lua = [0.43, 0.54, 0.6, 0.52, 0.6]
no_sandbox_contiguous_2000_py = [1.51, 1.45, 1.54, 1.69, 1.47]

no_sandbox_chunked_4000_cpp = [4.78, 4.83, 4.93, 4.76, 4.91]
no_sandbox_chunked_4000_cuda_1 = [4.45, 4.47, 4.35, 4.35, 4.65]
no_sandbox_chunked_4000_cuda_16 = [1.96, 1.99, 2.01, 2.0, 2.0]
no_sandbox_chunked_4000_cuda_2 = [3.37, 3.49, 3.46, 3.32, 3.52]
no_sandbox_chunked_4000_cuda_32 = [2.4, 2.58, 2.38, 2.52, 2.41]
no_sandbox_chunked_4000_cuda_4 = [2.49, 2.55, 2.51, 2.54, 2.7]
no_sandbox_chunked_4000_cuda_8 = [2.1, 2.12, 2.07, 2.25, 2.12]
no_sandbox_chunked_4000_ds1 = [2.94, 2.95, 2.99, 2.95, 2.99]
no_sandbox_chunked_4000_lua = [4.71, 4.71, 4.69, 4.73, 4.88]
no_sandbox_chunked_4000_py = [8.24, 8.33, 8.37, 8.3, 8.28]

no_sandbox_contiguous_4000_cpp = [1.11, 1.29, 1.17, 1.13, 1.02]
no_sandbox_contiguous_4000_cuda_1 = [1.62, 1.58, 1.62, 1.7, 1.6]
no_sandbox_contiguous_4000_cuda_16 = [2.19, 2.3, 2.21, 2.21, 2.24]
no_sandbox_contiguous_4000_cuda_2 = [1.84, 2.13, 1.91, 1.83, 1.98]
no_sandbox_contiguous_4000_cuda_32 = [2.18, 2.15, 2.18, 2.16, 2.17]
no_sandbox_contiguous_4000_cuda_4 = [2.5, 2.6, 2.51, 2.54, 2.5]
no_sandbox_contiguous_4000_cuda_8 = [2.18, 2.24, 2.26, 2.26, 2.17]
no_sandbox_contiguous_4000_ds1 = [0.6, 0.48, 0.64, 0.59, 0.55]
no_sandbox_contiguous_4000_lua = [1.21, 1.05, 1.07, 1.18, 1.07]
no_sandbox_contiguous_4000_py = [4.68, 4.66, 4.68, 4.67, 4.44]

no_sandbox_chunked_8000_cpp = [9.16, 9.14, 9.29, 9.35, 9.15]
no_sandbox_chunked_8000_cuda_1 = [7.92, 7.91, 7.82, 7.97, 7.88]
no_sandbox_chunked_8000_cuda_16 = [3.4, 3.36, 3.43, 3.38, 3.38]
no_sandbox_chunked_8000_cuda_2 = [6.45, 6.27, 6.31, 6.47, 6.39]
no_sandbox_chunked_8000_cuda_32 = [3.73, 3.83, 4.72, 3.88, 4.08]
no_sandbox_chunked_8000_cuda_4 = [4.32, 4.29, 4.33, 4.32, 4.32]
no_sandbox_chunked_8000_cuda_8 = [3.48, 3.4, 3.48, 3.46, 3.43]
no_sandbox_chunked_8000_ds1 = [4.89, 4.86, 4.85, 4.82, 4.85]
no_sandbox_chunked_8000_lua = [9.24, 9.29, 9.34, 9.09, 9.15]
no_sandbox_chunked_8000_py = [22.32, 22.26, 22.47, 22.48, 22.52]

no_sandbox_contiguous_8000_cpp = [3.68, 3.64, 3.74, 3.73, 3.57]
no_sandbox_contiguous_8000_cuda_1 = [1.78, 1.89, 1.83, 1.85, 1.8]
no_sandbox_contiguous_8000_cuda_16 = [4.44, 4.51, 4.25, 4.43, 4.46]
no_sandbox_contiguous_8000_cuda_2 = [3.15, 3.42, 3.49, 3.15, 3.46]
no_sandbox_contiguous_8000_cuda_32 = [4.34, 4.33, 4.3, 4.32, 4.25]
no_sandbox_contiguous_8000_cuda_4 = [5.71, 5.67, 6.04, 7.16, 5.99]
no_sandbox_contiguous_8000_cuda_8 = [4.87, 4.87, 4.93, 4.63, 4.82]
no_sandbox_contiguous_8000_ds1 = [1.63, 1.4, 1.77, 1.27, 1.38]
no_sandbox_contiguous_8000_lua = [3.51, 3.8, 3.66, 3.74, 3.07]
no_sandbox_contiguous_8000_py = [16.6, 16.68, 16.63, 16.7, 16.22]


myself = sys.modules[__name__]
modes = ["contiguous", "chunked"]
### grid_sizes = ['1000', '16000', '2000', '4000', '8000']
grid_sizes = ["1000", "2000", "4000", "8000", "16000"]

def compare_lang(entry):
  # langs = ['cpp', 'lua', 'py', 'cuda_1', 'cuda_16', 'cuda_2', 'cuda_32', 'cuda_4', 'cuda_8']
  return int(entry.split("_")[1]) if entry.find("_") > 0 else ord(entry[0])*100 + ord(entry[1])

sorted_langs = ['cpp', 'ds1', 'lua', 'py', 'cuda_1', 'cuda_16', 'cuda_2', 'cuda_32', 'cuda_4', 'cuda_8']
sorted_langs.sort(key=compare_lang)
sorted_langs = ['cuda_1', 'cuda_2', 'cuda_4', 'cuda_8', 'cuda_16', 'cuda_32', 'cpp', 'lua', 'py', 'ds1']
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



