import os
import pickle
import numpy as np
from .datasets import HeatmapDataset


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def load_heatmaps(filename, symmetric=True):
    if filename is None:
        return None
    heatmaps, *_ = load_dataset(filename)
    if (heatmaps >= 0).all():
        print("Warning: heatmaps where not stored in logaritmic space, conversion may be lossy!")
        heatmaps = np.log(heatmaps)
    return heatmaps if not symmetric else np.maximum(heatmaps, np.transpose(heatmaps, (0, 2, 1)))


def unpack_heatmaps(batch):
    if isinstance(batch, dict) and 'heatmap' in batch and 'data' in batch:
        return batch['data'], batch['heatmap']
    return batch, None


def pack_heatmaps(dataset, opts, offset=None):
    if opts.heatmap is None:
        return dataset
    offset = offset or opts.offset
    return HeatmapDataset(dataset, load_heatmaps(opts.heatmap, symmetric=True)[offset:offset+len(dataset)])


def outfile_name(data_path, beam, args):
    " Create unique output file name based on arguments "
    dataset_basename, ext = os.path.splitext(os.path.split(data_path)[-1]) 
    heatmap_basename, _ = os.path.splitext(os.path.split(args.heatmap)[-1]) if args.heatmap is not None else ""   
    results_dir = os.path.join(args.results_dir, 'tsp', dataset_basename)
    outfile = os.path.join(results_dir, "{}{}{}-{}-{}{}{}-{}{}{}".format(
        dataset_basename,
        "offs{}".format(args.offset) if args.offset is not None else "",
        "n{}".format(args.val_size) if args.val_size is not None else "",
        heatmap_basename,
        args.decode_strategy, beam, args.score_function,
        "th" + str(args.heatmap_threshold) if args.heatmap_threshold is not None else "",
        "knn" + str(args.knn) if args.knn is not None else "",
        ext
    ))
    return outfile
