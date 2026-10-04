from enum import Enum

import numpy as np
import numpy.typing as npt


class Device(Enum):
    CPU = 0
    GPU = 1


class Backend:
    @property
    def xp(self) -> np:
        raise NotImplementedError

    @property
    def device(self) -> Device:
        raise NotImplementedError

    def as_numpy(self, array: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def scatter_add(
        self, array: npt.NDArray, indices: npt.NDArray, values: npt.NDArray
    ):
        raise NotImplementedError


class NumpyBackend(Backend):
    @property
    def xp(self):
        return np

    @property
    def device(self):
        return Device.CPU

    def as_numpy(self, array: npt.NDArray) -> npt.NDArray:
        return array

    def scatter_add(
        self, array: npt.NDArray, indices: npt.NDArray, values: npt.NDArray
    ):
        np.add.at(array, indices, values)


class CudaBackend(Backend):
    def __init__(self):
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No CUDA device found")

    @property
    def xp(self):
        import cupy

        return cupy

    @property
    def device(self):
        return Device.GPU

    def as_numpy(self, array: npt.NDArray) -> npt.NDArray:
        return self.xp.asnumpy(array)

    def scatter_add(
        self, array: npt.NDArray, indices: npt.NDArray, values: npt.NDArray
    ):
        import cupyx

        cupyx.scatter_add(array, indices, values)


def get_backend(device: Device):
    if device == Device.CPU:
        return NumpyBackend()
    else:
        return CudaBackend()
