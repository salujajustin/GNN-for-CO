import os
import pickle
import numpy as np


# Code adapted from https://github.com/wouterkool/dpdp
def print_statistics(results, opts):
    num_processes = opts.system_info['used_num_processes']
    device_count = opts.system_info['used_device_count']
    batch_size = opts.batch_size
    assert num_processes % device_count == 0
    num_processes_per_device = num_processes // device_count

    results_stat = [(cost, tour, duration) for (cost, tour, duration) in results if tour is not None]
    if len(results_stat) < len(results):
        failed = [i + opts.offset for i, (cost, tour, duration) in enumerate(results) if tour is None]
        print("*" * 100)
        print("FAILED {} of {} instances, only showing statistics for {} solved instances!".format(
            len(results) - len(results_stat), len(results), len(results_stat)))
        print("Instances failed (showing max 10): ", failed[:10])
        print("*" * 100)
        # results = results_stat
    costs, tours, durations = zip(*results_stat)  # Not really costs since they should be negative
    print("Costs (showing max 10): ", costs[:10])
    if len(tours) == 1:
        print("Tour", tours[0])
    print("Average cost: {:.3f} +- {:.3f}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))

    avg_serial_duration, avg_parallel_duration, total_duration_parallel, total_duration_single_device, effective_batch_size = get_durations(
        durations, batch_size, num_processes, device_count
    )

    print("Average serial duration (per process per device): {:.3f}".format(avg_serial_duration))
    if batch_size > 1:
        print("Average parallel duration (per process per device), effective batch size {:.2f}): {:.3f}".format(
            effective_batch_size, avg_parallel_duration))
    if device_count > 1:
        print(
            "Calculated total duration for {} instances with {} processes x {} devices (= {} proc) in parallel: {}".format(
                len(durations), num_processes_per_device, device_count, num_processes, total_duration_parallel))
    # On 1 device it takes k times longer than on k devices
    print("Calculated total duration for {} instances with {} processes on 1 device in parallel: {}".format(
        len(durations), num_processes_per_device, total_duration_single_device))
    print("Number of GPUs used:", device_count)
    return costs, durations, tours

