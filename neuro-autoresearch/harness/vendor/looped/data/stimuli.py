"""Vendored looped-networks lazy stimulus readers; model preprocessing is adapter-owned."""
import io

import h5py
import numpy as np
from PIL import Image


class LazyHdf5Dict:
    """Read-only dict {str(stimulus id): image} that reads each image from the HDF5 file on demand."""

    def __init__(self, hdf5_path: str, dataset_name: str, indices):
        self._hdf5_path = hdf5_path
        self._dataset_name = dataset_name
        self._index_map = {str(idx): int(idx) for idx in indices}
        self._keys_sorted = sorted(self._index_map.keys(), key=lambda x: int(x))
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        return self._file

    def __contains__(self, key):
        return str(key) in self._index_map

    def __len__(self):
        return len(self._index_map)

    def keys(self):
        return self._keys_sorted

    def __getitem__(self, key):
        key_str = str(key)
        if key_str not in self._index_map:
            raise KeyError(key)
        return self._open()[self._dataset_name][self._index_map[key_str]]

    def __del__(self):
        if self._file is not None:
            self._file.close()


class LazyEncodedHdf5Dict:
    """Read-only dict {str(stimulus id): uint8 RGB image} decoding each encoded image of the HDF5 file on demand."""

    def __init__(self, hdf5_path, indices):
        self._hdf5_path = hdf5_path
        self._keys_sorted = sorted((str(int(i)) for i in indices), key=int)
        self._keys = set(self._keys_sorted)
        self._file = None

    def __contains__(self, key):
        return str(key) in self._keys

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys_sorted

    def __getitem__(self, key):
        if str(key) not in self._keys:
            raise KeyError(key)
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        im = Image.open(io.BytesIO(bytes(self._file["images"][int(key)])))
        if im.mode in ("RGBA", "LA", "P"):  # 131 OOD PNGs: shown alpha-composited over PsychoPy's mid-grey
            im = Image.alpha_composite(Image.new("RGBA", im.size, (128, 128, 128, 255)), im.convert("RGBA"))
        return np.asarray(im.convert("RGB"))

    def __getstate__(self):  # never hand an open h5 handle to a DataLoader worker
        return {**self.__dict__, "_file": None}

    def __del__(self):
        if self._file is not None:
            self._file.close()
