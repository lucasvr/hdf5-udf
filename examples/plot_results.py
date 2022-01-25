#!/usr/bin/env python3

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import sys

no_sandbox_chunked_1000_cpp = [0.236509, 0.22558, 0.289703, 0.251608, 0.25612100000000004]
no_sandbox_chunked_1000_cuda_1 = [0.16958499999999999, 0.21559099999999998, 0.491509, 0.097382, 0.261922, 0.233157, 0.590347, 0.557555, 0.678635, 0.555317]
no_sandbox_chunked_1000_cuda_16 = [0.09837, 0.17675000000000002, 0.175304, 0.148043, 0.146951, 0.087448, 0.546928, 0.574212, 0.196404]
no_sandbox_chunked_1000_cuda_2 = [0.228185, 0.242983, 0.530466, 0.095949, 0.091491, 0.49094000000000004, 0.244548, 0.697033, 0.11421100000000001, 0.342912]
no_sandbox_chunked_1000_cuda_4 = [0.432168, 0.363969, 0.086593, 0.08965899999999999, 0.115183, 0.108127, 0.09695100000000001, 0.13342, 0.14318599999999998, 0.148333]
no_sandbox_chunked_1000_cuda_8 = [0.694861, 0.095288, 0.152336, 0.39144, 0.25576, 0.198016, 0.636252, 0.485186]
no_sandbox_chunked_1000_lua = [0.228615, 0.35192199999999996, 0.23178500000000002, 0.263016, 0.227848]
no_sandbox_chunked_1000_py = [0.25388900000000003, 0.250587, 0.325785, 0.318398, 0.274791]

no_sandbox_contiguous_1000_cpp = [0.006699, 0.006713, 0.006732999999999999, 0.00675, 0.006683]
no_sandbox_contiguous_1000_cuda_1 = [0.005943, 0.005925, 0.0061530000000000005, 0.006184, 0.00621, 0.006174, 0.005933, 0.010936, 0.005911, 0.0059310000000000005]
no_sandbox_contiguous_1000_cuda_16 = [0.011207, 0.009142, 0.017551, 0.009543999999999999, 0.008081, 0.011995, 0.006412999999999999, 0.006162, 0.012167, 0.01048]
no_sandbox_contiguous_1000_cuda_2 = [0.004887, 0.004901, 0.005279, 0.0049, 0.004766, 0.004624, 0.004647, 0.006626999999999999, 0.006822, 0.004766]
no_sandbox_contiguous_1000_cuda_4 = [0.0061790000000000005, 0.006116, 0.005501000000000001, 0.006411, 0.00657, 0.014983, 0.009414, 0.0059570000000000005, 0.005549, 0.0054789999999999995]
no_sandbox_contiguous_1000_cuda_8 = [0.007032, 0.006104, 0.006031, 0.005971, 0.008079, 0.008048999999999999, 0.012314, 0.006889999999999999, 0.0061129999999999995]
no_sandbox_contiguous_1000_lua = [0.006702, 0.006606, 0.006706, 0.006607, 0.0066359999999999995]
no_sandbox_contiguous_1000_py = [0.006727, 0.006839, 0.006676, 0.0067410000000000005, 0.0067]

no_sandbox_chunked_16000_cpp = [4.112226, 4.076969, 4.313478, 4.073392, 4.083839]
no_sandbox_chunked_16000_cuda_1 = [2.180009, 1.816026, 1.637639, 1.676422, 1.941423, 1.811977, 2.788564, 1.797053, 1.8953849999999999, 2.1710000000000003]
no_sandbox_chunked_16000_cuda_16 = [1.5312260000000002, 1.2687439999999999, 1.3098839999999998, 1.271706, 1.4616859999999998, 1.480245, 1.559605, 1.820708]
no_sandbox_chunked_16000_cuda_2 = [1.490861, 1.2621030000000002, 1.123727, 1.174931, 1.27794, 1.517523, 1.1844869999999998, 1.170717, 1.579281, 1.2340010000000001]
no_sandbox_chunked_16000_cuda_4 = [1.535116, 1.3352300000000001, 1.39038, 1.316497, 1.3001040000000001, 1.406573, 1.691303, 1.817221, 1.898228, 1.318437]
no_sandbox_chunked_16000_cuda_8 = [1.680438, 1.9864860000000002, 1.263584, 1.257402, 1.6341679999999998, 2.003451, 1.329957]
no_sandbox_chunked_16000_lua = [4.705659000000001, 4.155164, 4.082904, 4.118945, 4.320489]
no_sandbox_chunked_16000_py = [4.413279, 4.150151, 4.630571, 4.277919000000001, 4.171867000000001]

no_sandbox_contiguous_16000_cpp = [1.4575390000000001, 1.470341, 1.459146, 1.453629, 1.455227]
no_sandbox_contiguous_16000_cuda_1 = [1.031293, 1.035064, 1.028493, 1.0554109999999999, 1.034839, 1.029284, 1.030675, 1.035989, 1.034988, 1.0366469999999999]
no_sandbox_contiguous_16000_cuda_16 = [0.30984500000000004, 0.31638299999999997, 0.307893, 0.30276899999999995, 0.309219, 0.312183, 0.306883, 0.309678]
no_sandbox_contiguous_16000_cuda_2 = [0.554902, 0.552448, 0.552834, 0.552211, 0.550629, 0.6013660000000001, 0.551311, 0.5634600000000001, 0.551656, 0.5562199999999999]
no_sandbox_contiguous_16000_cuda_4 = [0.32749, 0.331766, 0.344569, 0.33328, 0.342314, 0.32986, 0.3409, 0.33452, 0.345908, 0.33040899999999995]
no_sandbox_contiguous_16000_cuda_8 = [0.30204600000000004, 0.30468700000000004, 0.30116699999999996, 0.303129, 0.3028, 0.303427, 0.301824, 0.302277, 0.30260200000000004, 0.302693]
no_sandbox_contiguous_16000_lua = [1.460403, 1.456509, 1.455672, 1.4598300000000002, 1.4640909999999998]
no_sandbox_contiguous_16000_py = [1.4693239999999999, 1.456445, 1.457772, 1.456092, 1.478497]

no_sandbox_chunked_2000_cpp = [0.41571199999999997, 0.606501, 0.386571, 0.351377, 0.330309]
no_sandbox_chunked_2000_cuda_1 = [0.5451870000000001, 0.557191, 0.147241, 0.543953, 0.704971, 0.17083700000000002, 0.160948, 0.237117, 0.22786800000000001, 0.311015]
no_sandbox_chunked_2000_cuda_16 = [0.103364, 0.419631, 0.43377699999999997, 0.485627, 0.6199319999999999, 0.168174, 0.6124529999999999, 0.333326, 0.141592]
no_sandbox_chunked_2000_cuda_2 = [0.145829, 0.15178, 0.108462, 0.10694400000000001, 0.519845, 0.562743, 0.12842900000000002, 0.6419710000000001, 0.15865, 0.548657]
no_sandbox_chunked_2000_cuda_4 = [0.632035, 0.192275, 0.19166699999999998, 0.088644, 0.091645, 0.090904, 0.094937, 0.106377, 0.099971, 0.563957]
no_sandbox_chunked_2000_cuda_8 = [0.5119859999999999, 0.277929, 0.516121, 0.609599, 0.633132, 0.475769, 0.602358, 0.09056800000000001, 0.090775, 0.115165]
no_sandbox_chunked_2000_lua = [0.345001, 0.376458, 0.310053, 0.348823, 0.3445]
no_sandbox_chunked_2000_py = [0.40198199999999995, 0.437727, 0.441676, 0.401612, 0.36154699999999995]

no_sandbox_contiguous_2000_cpp = [0.025474999999999998, 0.025365, 0.025354, 0.025492, 0.025438000000000002]
no_sandbox_contiguous_2000_cuda_1 = [0.019611, 0.019835, 0.019734, 0.019826, 0.019644000000000002, 0.019616, 0.019563999999999998, 0.019636, 0.019754, 0.019537]
no_sandbox_contiguous_2000_cuda_16 = [0.016994, 0.017473000000000002, 0.020617, 0.019812, 0.020401, 0.024703]
no_sandbox_contiguous_2000_cuda_2 = [0.011914999999999999, 0.012282000000000001, 0.012427, 0.012361, 0.012007, 0.012502, 0.012279, 0.012761, 0.015171, 0.012655]
no_sandbox_contiguous_2000_cuda_4 = [0.013528, 0.011126, 0.010705, 0.0105, 0.012279, 0.01044, 0.009775, 0.013296, 0.010225, 0.010138]
no_sandbox_contiguous_2000_cuda_8 = [0.01341, 0.012412, 0.012194, 0.012004999999999998, 0.011349999999999999, 0.011272]
no_sandbox_contiguous_2000_lua = [0.025417000000000002, 0.025568, 0.025356, 0.025629, 0.025480000000000003]
no_sandbox_contiguous_2000_py = [0.025386, 0.025596, 0.025717999999999998, 0.025695000000000003, 0.025816]

no_sandbox_chunked_32000_cpp = [14.706028, 14.551556000000001, 14.646327, 14.621171, 14.650965]
no_sandbox_chunked_32000_cuda_1 = [6.653397999999999, 5.764632, 6.08323, 7.344710000000001, 6.330004, 5.75861, 5.7879830000000005, 5.880152, 6.037440999999999, 6.809621]
no_sandbox_chunked_32000_cuda_16 = [5.517619, 4.799684, 5.505635, 5.090278, 4.594202, 4.967406, 4.985561000000001, 4.5930420000000005, 5.5861909999999995]
no_sandbox_chunked_32000_cuda_2 = [4.600467999999999, 4.800295, 4.7234359999999995, 4.33528, 4.5105640000000005, 4.3719850000000005, 4.503484, 4.6981649999999995, 4.371747, 4.474183999999999]
no_sandbox_chunked_32000_cuda_4 = [4.898463, 5.275019, 4.503787, 5.546903, 5.028785, 5.523626, 4.618001, 4.634834, 4.283517, 4.906701]
no_sandbox_chunked_32000_cuda_8 = [6.140888, 5.147411, 5.2967960000000005, 4.608539, 4.931357, 6.436372, 5.059301, 4.859883, 4.6498360000000005]
no_sandbox_chunked_32000_lua = [14.727488, 15.067668999999999, 14.681845, 14.803894, 15.156142]
no_sandbox_chunked_32000_py = [15.483766, 14.922236, 14.713097999999999, 15.416592, 14.60182]

no_sandbox_contiguous_32000_cpp = [5.712491, 5.751061, 5.753069999999999, 5.790661999999999, 5.775446]
no_sandbox_contiguous_32000_cuda_1 = [4.516106000000001, 4.514856, 4.516733, 4.522085000000001, 4.521511, 4.525179, 4.507319, 4.561828, 4.538195, 4.5122800000000005]
no_sandbox_contiguous_32000_cuda_16 = [1.193949, 1.188087, 1.189168, 1.204904, 1.187878]
no_sandbox_contiguous_32000_cuda_2 = [2.375387, 2.383123, 2.370953, 2.429061, 2.3790459999999998, 2.381234, 2.375997, 2.3761919999999996, 2.376582, 2.3859269999999997]
no_sandbox_contiguous_32000_cuda_4 = [1.37196, 1.361768, 1.344605, 1.341858, 1.3543180000000001, 1.397233, 1.372681, 1.371523, 1.3991790000000002, 1.37982]
no_sandbox_contiguous_32000_cuda_8 = [1.185024, 1.182701, 1.184138, 1.185911, 1.212083, 1.184388, 1.190181]
no_sandbox_contiguous_32000_lua = [5.784054, 5.830126, 5.731325999999999, 5.722695, 5.712829]
no_sandbox_contiguous_32000_py = [6.189953, 5.750316, 5.8443760000000005, 5.770025, 5.720892]

no_sandbox_chunked_4000_cpp = [0.655284, 0.60745, 0.7963749999999999, 0.583712, 0.619326]
no_sandbox_chunked_4000_cuda_1 = [0.337683, 0.274014, 0.327608, 0.280941, 0.479826, 0.353749, 0.785746, 0.608915, 0.567993, 0.508272]
no_sandbox_chunked_4000_cuda_16 = [0.6654249999999999, 0.600678, 0.175836, 0.167425, 0.166522, 0.456322, 0.168387, 0.15538400000000002, 0.15889399999999998, 0.174627]
no_sandbox_chunked_4000_cuda_2 = [0.254901, 0.523967, 0.551582, 0.6216839999999999, 0.21726499999999999, 0.19169000000000003, 0.20167400000000002, 0.19683099999999998, 0.18714, 0.182499]
no_sandbox_chunked_4000_cuda_4 = [0.545872, 0.652517, 0.640977, 0.661818, 0.260563, 0.54231, 0.5465599999999999, 0.276304, 0.169509, 0.526036]
no_sandbox_chunked_4000_cuda_8 = [0.215397, 0.18968800000000002, 0.174566, 0.15298699999999998, 0.17991400000000002, 0.294766, 0.624186, 0.38322, 0.21607300000000002]
no_sandbox_chunked_4000_lua = [0.61027, 0.559226, 0.587345, 0.574163, 0.564792]
no_sandbox_chunked_4000_py = [0.6270070000000001, 0.676888, 0.618227, 0.635647, 0.63903]

no_sandbox_contiguous_4000_cpp = [0.09740399999999999, 0.09798499999999999, 0.097604, 0.097501, 0.098502]
no_sandbox_contiguous_4000_cuda_1 = [0.08085400000000001, 0.08107500000000001, 0.080882, 0.082951, 0.080894, 0.083919, 0.080927, 0.080946, 0.081397, 0.081021]
no_sandbox_contiguous_4000_cuda_16 = [0.032181, 0.031982, 0.037968, 0.040031, 0.029964]
no_sandbox_contiguous_4000_cuda_2 = [0.045441999999999996, 0.04503, 0.04474, 0.046979999999999994, 0.044542, 0.044650999999999996, 0.0441, 0.044301999999999994, 0.044412999999999994, 0.044020000000000004]
no_sandbox_contiguous_4000_cuda_4 = [0.032714, 0.029658, 0.031671000000000005, 0.02998, 0.031128999999999997, 0.030018, 0.030511, 0.031163999999999997, 0.031946999999999996, 0.031292]
no_sandbox_contiguous_4000_cuda_8 = [0.027610999999999997, 0.027361, 0.028283000000000003, 0.027354999999999997, 0.028575999999999997]
no_sandbox_contiguous_4000_lua = [0.09772700000000001, 0.097993, 0.098535, 0.09800400000000001, 0.09839300000000001]
no_sandbox_contiguous_4000_py = [0.098369, 0.097426, 0.097621, 0.098997, 0.098887]

no_sandbox_chunked_8000_cpp = [1.360589, 1.343597, 1.552344, 1.6069900000000001, 1.587836]
no_sandbox_chunked_8000_cuda_1 = [0.675871, 0.8278570000000001, 1.1822780000000002, 1.187132, 1.054211, 0.899726, 1.165282, 0.6391290000000001, 0.591305, 0.6144750000000001]
no_sandbox_chunked_8000_cuda_16 = [0.39767399999999997, 0.368919, 0.620943, 0.7909809999999999, 0.801625, 0.431358, 0.797255]
no_sandbox_chunked_8000_cuda_2 = [0.591373, 0.46801000000000004, 0.392009, 0.364925, 0.35688200000000003, 0.393534, 0.40230299999999997, 0.517977, 0.7670189999999999, 0.41891100000000003]
no_sandbox_chunked_8000_cuda_4 = [0.5745439999999999, 1.003452, 0.760113, 0.472399, 0.869774, 0.802052, 0.450324, 0.405862, 0.424228, 0.427829]
no_sandbox_chunked_8000_cuda_8 = [0.672864, 0.6173059999999999, 0.610222, 0.534609, 0.5131140000000001, 0.605352, 0.40942500000000004, 0.512205]
no_sandbox_chunked_8000_lua = [1.404798, 1.412299, 1.434145, 1.422744, 1.352873]
no_sandbox_chunked_8000_py = [1.724008, 1.3917359999999999, 1.361732, 1.4483570000000001, 1.618507]

no_sandbox_contiguous_8000_cpp = [0.373011, 0.372617, 0.374029, 0.373518, 0.37427699999999997]
no_sandbox_contiguous_8000_cuda_1 = [0.267272, 0.267319, 0.26809700000000003, 0.270343, 0.266806, 0.267645, 0.26768400000000003, 0.268238, 0.26713699999999996, 0.271521]
no_sandbox_contiguous_8000_cuda_16 = [0.088034, 0.08998400000000001, 0.083925, 0.086852, 0.088895, 0.089858, 0.083947, 0.08676800000000001, 0.121635, 0.089932]
no_sandbox_contiguous_8000_cuda_2 = [0.14814, 0.14846900000000002, 0.148553, 0.148589, 0.148001, 0.14771800000000002, 0.151704, 0.149295, 0.147979, 0.14809899999999998]
no_sandbox_contiguous_8000_cuda_4 = [0.089158, 0.08863599999999999, 0.090193, 0.092055, 0.091995, 0.091092, 0.091903, 0.09697800000000001, 0.088727, 0.091452]
no_sandbox_contiguous_8000_cuda_8 = [0.08110200000000001, 0.08110300000000001, 0.083186, 0.081676, 0.082997, 0.081678, 0.104203, 0.080823]
no_sandbox_contiguous_8000_lua = [0.374112, 0.37407, 0.374587, 0.373274, 0.37351100000000004]
no_sandbox_contiguous_8000_py = [0.372997, 0.374516, 0.374459, 0.37336499999999995, 0.374329]


myself = sys.modules[__name__]
modes = ["contiguous", "chunked"]
grid_sizes = ['1000', '2000', '4000', '8000', '16000', '32000']

def compare_lang(entry):
  # langs = ['cpp', 'lua', 'py', 'cuda_1', 'cuda_16', 'cuda_2', 'cuda_32', 'cuda_4', 'cuda_8']
  return int(entry.split("_")[1]) if entry.find("_") > 0 else ord(entry[0])*100 + ord(entry[1])

sorted_langs = ['cpp', 'cuda_1', 'cuda_16', 'cuda_2', 'cuda_4', 'cuda_8', 'lua', 'py']
sorted_langs.sort(key=compare_lang)
sorted_langs = ['cuda_1', 'cuda_2', 'cuda_4', 'cuda_8', 'cuda_16', 'cpp', 'lua', 'py']

fmt_styles = []
pretty_names = []
for lang in sorted_langs:
    lang = lang.replace('cpp', 'C++').replace('lua', 'LuaJIT').replace('py', 'CPython').replace('ds1', 'Reference')
    if lang.startswith('cuda'):
        n = int(lang.split('_')[1])
        stream = 'stream' if n == 1 else 'streams'
        lang = 'NVIDIA GDS ({} {})'.format(n, stream)
    pretty_names.append(lang)
    fmt_styles.append('-o' if lang.startswith('NVIDIA') else '--o')

print("No-sandbox (GDS)")
        
# No-sandbox
fig = plt.figure(figsize=(8.5, 4.5), constrained_layout=True)
gs = fig.add_gridspec(1, 2, hspace=1, wspace=0.05)
axs = gs.subplots(sharex=False, sharey=True)
for i, mode in enumerate(modes):
    this_plt = axs[i]
    for j, lang in enumerate(sorted_langs):
        label = pretty_names[j]
        datapoints_avg = [np.average(getattr(myself, "no_sandbox_" + mode + "_" + grid + "_" + lang)) for grid in grid_sizes]
        datapoints_std = [np.std(getattr(myself, "no_sandbox_" + mode + "_" + grid + "_" + lang)) for grid in grid_sizes]
        
        this_plt.plot(['1000²', '2000²', '4000²', '8000²', '16000²', '32000²'], datapoints_avg, fmt_styles[j], label=label)
        #this_plt.errorbar(['1000²', '2000²', '4000²', '8000²', '16000²', '32000²'], datapoints_avg, yerr=datapoints_std, fmt=fmt_styles[j], label=label)

    mode = mode[0].upper() + mode[1:]
    if i == 0:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

        this_plt.set_ylabel("Time (secs)", fontname='Times New Roman', fontsize=10)
        this_plt.legend(loc="upper left")
        fig.supxlabel("Dataset size", fontname='Times New Roman', fontsize=10)
        this_plt.set_title("{} datasets".format(mode), fontname='Times New Roman', fontsize=11)
    else:
        this_plt.set_title("{} datasets + GPU-side decompression".format(mode), fontname='Times New Roman', fontsize=11)

    this_plt.set_yscale("log")

for ax in fig.get_axes():
    ax.label_outer()

plt.savefig("perf-no_sandbox-logscale.pdf")
#plt.cla()
#plt.clf()
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



