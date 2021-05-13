import sys
import os
import argparse
from os import path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Batch size to use during evaluation (per GPU)")
    parser.add_argument('--beam_size', type=int, nargs='+',
                        help='Sizes of beam to use for beam search/DP')
    parser.add_argument('--decode_strategy', type=str,
                        help='Deep Policy Dynamic Programming (dpdp) or Deep Policy Beam Search (dpbs)')
    parser.add_argument('--score_function', type=str, default='model_local',
                        help="Policy/score function to use to select beam: 'cost', 'heatmap' or 'heatmap_potential'")
    parser.add_argument('--problem', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--verbose', action='store_true', help='Set to show statistics')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use per device (cpu or gpu).')
    # When providing a heatmap, will sparsify the input
    parser.add_argument('--heatmap', default=None, help="Heatmaps to use")
    parser.add_argument('--heatmap_threshold', type=float, default=None, help="Use sparse graph based on heatmap treshold")
    parser.add_argument('--knn', type=int, default=None, help="Use sparse knn graph")
    parser.add_argument('--kthvalue_method', type=str, default='sort', help="Which kthvalue method to use for dpdp ('auto' = auto determine)")

    args = parser.parse_args()
    args.beam_size = args.beam_size if args.beam_size is not None else [0]
    assert args.o is None or (len(args.datasets) == 1 and len(args.beam_size) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one beam_size"
    assert args.heatmap is None or len(args.datasets) == 1, "With heatmap can only run one (corresponding) dataset"

    return args
