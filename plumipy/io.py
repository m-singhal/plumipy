import h5py
import numpy as np

def load_hdf5_results(filename):
    def load_group(group):
        data = {}
        for key in group.keys():
            item = group[key]

            if isinstance(item, h5py.Group):
                data[key] = load_group(item)
            else:
                data[key] = item[()]

        return data

    with h5py.File(filename, "r") as f:
        return load_group(f)