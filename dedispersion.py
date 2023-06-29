#!/usr/bin/env python
import os
import numpy as np
from collections import OrderedDict
import kernel_tuner as kt
from kernel_tuner.observers.powersensor import PowerSensorObserver

nr_dms = 2048
nr_samples = 25000
nr_channels = 1536
down_sampling = 1
dm_first = 0.0
dm_step = 0.02

channel_bandwidth = 0.1953125
sampling_time = 0.00004096
min_freq = 1425.0
max_freq = min_freq + (nr_channels-1) * channel_bandwidth

HEADER_TEMPLATE = """
#define nr_dms {nr_dms}
#define nr_samples {nr_samples}
#define nr_channels {nr_channels}
#define max_shift {max_shift}
#define nr_samples_per_channel ({nr_samples} + {max_shift})
#define dm_first {dm_first}f
#define dm_step {dm_step}f
"""


def get_shift(freq):
    inverse_high_freq = max_freq**-2
    inverse_freq = freq**-2
    # 4148.808 is the time delay per dispersion measure, a constant in the dispersion equation
    shift = 4148.808 * (inverse_freq - inverse_high_freq) / (sampling_time * down_sampling)
    return shift


max_shift = int(np.ceil(get_shift(min_freq) * (dm_first + nr_dms * dm_step))) + 1
nr_samples_per_channel = (nr_samples+max_shift)


def get_shifts():
    channels = np.arange(nr_channels, dtype=np.float32)
    freqs = min_freq + (channels * channel_bandwidth)
    shifts_float = get_shift(freqs)
    shifts_float[-1] = 0
    return shifts_float


def write_header():
    header = HEADER_TEMPLATE.format(nr_dms=nr_dms,
                                    nr_samples=nr_samples,
                                    nr_channels=nr_channels,
                                    max_shift=max_shift,
                                    dm_first=dm_first,
                                    dm_step=dm_step)
    with open("dedispersion.h", "w") as fp:
        fp.write(header)


def create_reference():
    write_header()

    input_samples = np.random.randn(nr_samples_per_channel * nr_channels).astype(np.uint8)
    output_arr = np.zeros(nr_dms * nr_samples, dtype=np.float32)
    shifts = get_shifts()

    kernel_name = "dedispersion_reference"
    kernel_string = "dedispersion.cc"
    args = [input_samples, output_arr, shifts]

    reference = kt.run_kernel(kernel_name, kernel_string, 1, args, {}, lang="C")

    np.save("input_ref", input_samples, allow_pickle=False)
    np.save("shifts_ref", shifts, allow_pickle=False)
    np.save("dedisp_ref", reference[1], allow_pickle=False)


def tune():
    input_samples = np.load("input_ref.npy")
    output_arr = np.zeros(nr_dms * nr_samples, dtype=np.float32)
    shifts = np.load("shifts_ref.npy")

    # ensure consistency of the input files
    assert max_shift > (dm_first + nr_dms * dm_step) * shifts[0]

    args = [input_samples, output_arr, shifts]

    problem_size = (nr_samples, nr_dms, 1)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [1, 2, 4, 8] + [16*i for i in range(1,3)]
    tune_params["block_size_y"] = [8*i for i in range(4,33)]
    tune_params["block_size_z"] = [1]
    tune_params["tile_size_x"] = [i for i in range(1,5)]
    tune_params["tile_size_y"] = [i for i in range(1,9)]
    tune_params["tile_stride_x"] = [0, 1]
    tune_params["tile_stride_y"] = [0, 1]
    #tune_params["loop_unroll_factor_x"] = [0] #[i for i in range(1,max(tune_params["tile_size_x"]))]
    #tune_params["loop_unroll_factor_y"] = [0] #[i for i in range(1,max(tune_params["tile_size_y"]))]
    tune_params["loop_unroll_factor_channel"] = [0] #+ [i for i in range(1,nr_channels+1) if nr_channels % i == 0] #[i for i in range(nr_channels+1)]
    #tune_params["blocks_per_sm"] = [i for i in range(5)]

    print("Parameters:")
    [print(k,v) for k,v in tune_params.items()]

    cp = [f"-I{os.path.dirname(os.path.realpath(__file__))}"]

    check_block_size = "32 <= block_size_x * block_size_y <= 1024"
    #check_loop_x = "loop_unroll_factor_x <= tile_size_x and tile_size_x % loop_unroll_factor_x == 0"
    #check_loop_y = "loop_unroll_factor_y <= tile_size_y and tile_size_y % loop_unroll_factor_y == 0"
    #check_loop_channel = f"loop_unroll_factor_channel <= {nr_channels} and loop_unroll_factor_channel and {nr_channels} % loop_unroll_factor_channel == 0"

    check_tile_stride_x = "tile_size_x > 1 or tile_stride_x == 0"
    check_tile_stride_y = "tile_size_y > 1 or tile_stride_y == 0"

    config_valid = [check_block_size, check_tile_stride_x, check_tile_stride_y]

    metrics = OrderedDict()
    gbytes = (nr_dms * nr_samples * nr_channels)/1e9
    metrics["GB/s"] = lambda p: gbytes / (p['time'] / 1e3)

    # power measurement
    ps_observer = PowerSensorObserver(["ps_energy"])
    metrics["GBs/W"] = lambda p: gbytes / p["ps_energy"]
    metrics["GPU (W)"] = lambda p: p["ps_energy"] / (p["time"]/1e3)

    results, env = kt.tune_kernel("dedispersion_kernel", "dedispersion.cu", problem_size, args, tune_params,
                                  compiler_options=cp, restrictions=config_valid,
                                  cache="dedisp_cache.json", strategy="random_sample",
                                  metrics=metrics, observers=[ps_observer])



if __name__ == "__main__":
    print("Creating reference ...")
    create_reference()

    # print("Tuning ...")
    # tune()
