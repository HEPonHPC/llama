import numpy as np

# from typing_extensions import Protocol
from typing import Union, Tuple, List

import h5py
import boost_histogram as bh
import pandana

import enum

from abc import ABC, abstractmethod


class Kind(str, enum.Enum):
    COUNT = "COUNT"
    MEAN = "MEAN"


class Array:
    """Wraps a numpy array for data-parallel operations"""

    def __init__(self, data: np.array):
        self._data = data

    @property
    def data(self) -> np.array:
        return self._data

    @property
    def shape(self):
        return self.data.shape

    def sum(self, *args, **kwargs) -> Union[float, int]:
        # reduce sum?
        return self._data.sum(*args, **kwargs)

    def __getitem__(self, i) -> Union[float, int]:
        return self._data[i]

    def __setitem__(self, i, val) -> None:
        self._data[i] = val

    def __array__(self) -> np.array:
        return self._data

    def __truediv__(self, other: Union["array", float, int]):
        """Calculate element by element quotient"""
        if isinstance(other, array):
            return array(self.data / other.data)
        else:
            return array(self.data * other)

    def __rtruediv__(self, other: Union["array", float, int]):
        return self.__truediv__(other)

    def __mul__(self, other: Union["array", float, int]):
        """Calculate the dot product with other"""
        if isinstance(other, array):
            return array(self.data * other.data)
        else:
            return array(self.data * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other: Union["array", np.array]) -> "array":
        if isinstance(other, array):
            return np.matmul(self.data, other.data)
        else:
            return np.matmul(self.data, other)


# class SupportsArray(Protocol):
#    def __array__(self: "SupportsArray") -> float:
#        pass


class AxisFactory:
    defaults = {
        "growth": False,
        "circular": False,
        "overflow": True,
        "underflow": True,
    }

    @staticmethod
    def Regular(label, *args, **kwargs):
        return bh.axis.Regular(*args, metadata=label, **AxisFactory.defaults, **kwargs)

    @staticmethod
    def Variable(label, *args, **kwargs):
        return bh.axis.Variable(*args, metadata=label, **AxisFactory.defaults, **kwargs)

    @staticmethod
    def from_array(label, edges, axtype):
        if issubclass(axtype, bh.axis.Regular):
            return AxisFactory.Regular(
                label, bins=len(edges) - 1, start=edges[0], stop=edges[-1]
            )
        elif issubclass(axtype, bh.axis.Variable):
            return AxisFactory.Variable(label, edges)

    @staticmethod
    def saveto(axis, group, name):
        if axis is not None:
            d = group.create_dataset(name, data=axis.edges, compression="gzip")
            d.attrs["label"] = axis.metadata
            d.attrs["type"] = (
                "regular" if isinstance(axis, bh.axis.Regular) else "variable"
            )
            return d

    @staticmethod
    def loadfrom(group, name):
        if name in group:
            d = group.get(name)
            atype = d.attrs["type"]
            atype = bh.axis.Regular if atype == "regular" else bh.axis.Variable
            return AxisFactory.from_array(d.attrs["label"], d[:].flatten(), atype)


class HistProxy:
    def __init__(self, hist):
        self.hist = hist

    def __add__(self, other_proxy):
        """Support operative syntax of histogram, but do nothing"""
        pass

    def __sub__(self, other_proxy):
        """Support operative syntax of histogram, but do nothing"""
        pass

    def __mult__(self, other_proxy):
        """Support operative syntax of histogram, but do nothing"""
        pass

    def __truediv__(self, other_proxy):
        """Support operative syntax of histogram, but do nothing"""
        pass


class Histogram:
    """Histogram events as a function of some event property"""

    def __init__(
        self,
        xaxis,
        yaxis=None,
        zaxis=None,
    ):
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis

        self.ndim = 1
        if yaxis:
            self.ndim = self.ndim + 1
        if zaxis:
            self.ndim = self.ndim + 1

        if self.ndim == 1:
            axes = (self.xaxis,)
        elif self.ndim == 2:
            axes = (self.xaxis, self.yaxis)
        else:
            axes = (self.xaxis, self.yaxis, self.zaxis)

        self.bhist = bh.Histogram(*axes, storage=bh.storage.Weight())

        self.set_w2()

        self.root = 0
        self.proxy = HistProxy(self)

        self._kind = None

    def set_root(self, rank):
        self.root = rank

    def unravel_axes(self):
        return AxisFactory.Regular(
            ";".join([ax.metadata for ax in self.axes]),
            bins=self.bhist.size,
            start=0,
            stop=self.bhist.size,
        )

    def gather(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        contents = self.get_contents()
        squared_errors = self.get_errors() ** 2

        contents = comm.reduce(contents, MPI.SUM, root=self.root)
        squared_errors = comm.reduce(squared_errors, MPI.SUM, root=self.root)

        if comm.Get_rank() == self.root:
            result = Histogram.from_filled(
                contents=contents,
                errors=np.sqrt(squared_errors),
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )
        else:
            result = Histogram(*self.axes)

        return result

    def copy(self):
        return Histogram.filled_histogram(
            contents=self.get_contents(),
            errors=self.get_errors(),
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            zaxis=self.zaxis,
        )

    def set_w2(self):
        self.errors = np.sqrt(self.bhist.variances(flow=True))

    def fill(self, *args, **kwargs):
        self.bhist.fill(*args, **kwargs)
        self.set_w2()

    def _axes_consistent_with(self, other):
        return self.bhist.axes == other.bhist.axes

    def get_bin_centers(self, flow=False):
        if flow:

            def pad_with_flow(axis):
                centers = np.zeros(axis.extent)
                centers[0] = axis.edges[0] - axis.widths[0]
                centers[-1] = axis.edges[-1] + axis.widths[-1]
                centers[1:-1] = axis.centers
                return centers

            return [pad_with_flow(ax) for ax in self.bhist.axes]
        else:
            return [centers.flatten() for centers in self.bhist.axes.centers]

    def get_contents(self, flow=False):
        return self.bhist.values(flow=flow)

    def set_contents(self, contents):
        self.bhist.reset()
        flow = contents.shape == self.bhist.counts(flow=True).shape
        self.bhist.fill(
            *[a.flatten() for a in np.meshgrid(*self.get_bin_centers(flow))],
            weight=contents
        )

    def get_errors(self, flow=False):
        if flow:
            return self.errors
        else:
            return self.errors[1:-1]

    def set_errors(self, errors):
        self.errors = errors

    def __truediv__(self, other: Union["histogram", int, float]) -> "histogram":
        """If other is a histogram, do bin by bin division.

        Since a1 / a2 + b1 / b2 =/= (a1 + b1) / (a2 + b2)
        both histograms in this operation will be reduced then divided.

        If other is numeric, divide all contents by other

        Args:
            other (histogram | int | float)
        Returns:
            histogram : new histogram
        """
        if isinstance(other, Histogram):
            assert self._axes_consistent_with(other)

            quotient_contents = self.get_contents(flow=True) / other.get_contents(
                flow=True
            )
            quotient_errors = (
                np.sqrt(
                    (other.get_errors(flow=True) / other.get_contents(flow=True)) ** 2
                    + (self.get_errors(flow=True) / self.get_contents(flow=True)) ** 2
                )
                * quotient_contents
            )
            return Histogram.from_filled(
                contents=quotient_contents,
                errors=quotient_errors,
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )

        elif isinstance(other, float) | isinstance(other, int):
            return type(self).from_filled(
                contents=self.get_contents(flow=True) / other,
                errors=self.get_errors(flow=True) / other,
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )
        else:
            raise TypeError

    def __eq__(self, other: "histogram") -> bool:
        """Two histograms are equal if their contents are equal

        TODO: Does this need to compare the global histograms or local ones?
        """
        if isinstance(other, Histogram):
            return self.bhist == other.bhist
        else:
            return False

    def __mul__(self, other: Union["histogram", int, float]) -> "histogram":
        """If other is a histogram, do bin by bin multiplication.

        Since a1 / a2 + b1 / b2 =/= (a1 + b1) / (a2 + b2)
        both histograms in this operation will be reduced then divided.

        If other is numeric, divide all contents by other

        Args:
            other (histogram | int | float)
        Returns:
            histogram : new histogram
        """
        if isinstance(other, Histogram):
            assert self._axes_consistent_with(other)
            prod_contents = other.get_contents(flow=True) * self.get_contents(flow=True)
            prod_errors = (
                np.sqrt(
                    (other.get_errors(flow=True) / other.get_contents(flow=True)) ** 2
                    + (self.get_errors(flow=True) / self.get_contents(flow=True)) ** 2
                )
                * prod_contents
            )
            return type(self).from_filled(
                contents=prod_contents,
                errors=prod_errors,
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )

        elif isinstance(other, float) | isinstance(other, int):
            return type(self).from_filled(
                contents=self.get_contents(flow=True) * other,
                errors=self.get_errors(flow=True),
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )
        else:
            raise TypeError

    def __add__(self, other: Union["histogram", int, float]) -> "histogram":
        """If other is a histogram, do bin-by-bin addition.
        Since addition is commutative, no reductions occur

        If other is numeric, add all bin contents by other
        Args:
            other (histogram | int | float) : histogram or numeric to add to this histogram
        Returns:
            histogram : new histogram
        """
        if isinstance(other, Histogram):
            assert self._axes_consistent_with(other)
            summed_contents = other.get_contents(flow=True) + self.get_contents(
                flow=True
            )
            summed_errors = np.sqrt(
                other.get_errors(flow=True) ** 2 + self.get_errors(flow=True) ** 2
            )
            return type(self).from_filled(
                contents=summed_contents,
                errors=summed_errors,
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )

        elif isinstance(other, float) | isinstance(other, int):
            return type(self).from_filled(
                contents=self.get_contents(flow=True) + other,
                errors=self.get_errors(flow=True),
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )
        else:
            raise TypeError

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: "histogram") -> "histogram":
        """If other is a histogram, do bin-by-bin subtraction.
        Since addition is commutative, no reductions occur.


        If other is numeric, subtract all bin contents by other
        Args:
            other (histogram | int | float) : histogram or numeric to add to this histogram
        Returns:
            histogram : new histogram
        """
        if isinstance(other, Histogram):
            assert self._axes_consistent_with(other)
            summed_contents = other.get_contents(flow=True) - self.get_contents(
                flow=True
            )
            summed_errors = np.sqrt(
                other.get_errors(flow=True) ** 2 + self.get_errors(flow=True) ** 2
            )
            return type(self).from_filled(
                contents=summed_contents,
                errors=summed_errors,
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )

        elif isinstance(other, float) | isinstance(other, int):
            return type(self).from_filled(
                contents=self.get_contents(flow=True) - other,
                errors=self.get_errors(flow=True),
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )
        else:
            raise TypeError

    def __rsub__(self, other):
        return self.__sub__(other)

    def eval(self, to: int = 0):
        """Do any necessary MPI communications
        Args:
            to (int) : rank to send data to for aggregation. Can also be a 'all'
                       to broadcast data to all ranks after aggregation
        Returns:
            histogram : result of the evaluation
        """
        return self

    def sum_contents(self, flow=False):
        return self.get_contents(flow).sum()

    def sum_errors(self, flow=False):
        return np.sqrt((self.get_errors(flow) ** 2).sum())

    def __len__(self):
        return len(self.bins)

    @staticmethod
    def from_filled(*, contents, errors, xaxis, yaxis=None, zaxis=None):
        hist = Histogram(xaxis, yaxis, zaxis)
        hist.set_contents(contents)
        hist.set_errors(errors)
        return hist

    @property
    def kind(self):
        return self._kind or self.bhist.kind

    @kind.setter
    def kind(self, value):
        self._kind = value

    def values(self):
        return self.get_contents()

    def variances(self):
        return np.square(self.get_errors())

    def counts(self):
        return self.bhist.counts()

    @property
    def axes(self):
        return self.bhist.axes

    def saveto(self, filename_or_handle, group_name):
        from mpi4py import MPI

        if isinstance(filename_or_handle, str):
            filename_or_handle = h5py.File(filename_or_handle, "r+")

        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            hist = self.gather()
        else:
            hist = self

        if comm.Get_rank() == self.root:
            group = filename_or_handle.create_group(group_name)
            group.create_dataset(
                "contents", data=hist.get_contents(flow=True), compression="gzip"
            )
            group.create_dataset(
                "errors", data=hist.get_errors(flow=True), compression="gzip"
            )
            AxisFactory.saveto(hist.xaxis, group, "xaxis")
            AxisFactory.saveto(hist.yaxis, group, "yaxis")
            AxisFactory.saveto(hist.zaxis, group, "zaxis")

            return group

    @staticmethod
    def loadfrom(filename_or_handle, group_name, return_group=False):
        if isinstance(filename_or_handle, str):
            filename_or_handle = h5py.File(filename_or_handle, "r")
        group = filename_or_handle.get(group_name)
        contents = group.get("contents")[:]
        errors = group.get("errors")[:]

        xaxis = AxisFactory.loadfrom(group, "xaxis")
        yaxis = AxisFactory.loadfrom(group, "yaxis")
        zaxis = AxisFactory.loadfrom(group, "zaxis")

        hist = Histogram.from_filled(
            contents=contents, errors=errors, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis
        )
        if return_group:
            return hist, group
        else:
            return hist


class Spectrum(Histogram):
    def __init__(self, *args, exposure=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exposure = exposure

    def copy(self):
        return Spectrum.from_filled(
            contents=self.get_contents(),
            errors=self.get_errors(),
            exposure=self.exposure,
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            zaxis=self.zaxis,
        )

    def fill(self, data, exposure, weights=None):
        super().fill(data, weight=weights)
        self.exposure = exposure.sum()

    @staticmethod
    def from_pandana(pand, *args, **kwargs):
        assert isinstance(pand, pandana.core.spectrum.Spectrum)
        s = Spectrum(*args, **kwargs)
        s.fill(pand.df(), pand.POT(), pand.weight())
        return s

    def get_exposure(self):
        return self.exposure

    @staticmethod
    def from_filled(*, contents, errors, xaxis, yaxis=None, zaxis=None, exposure=None):
        spec = Spectrum(xaxis, yaxis, zaxis)
        spec.exposure = exposure
        spec.set_contents(contents)
        spec.set_errors(errors)
        return spec

    def scale_to_exposure(self, new_exposure):
        scale = new_exposure / self.exposure
        # make new spectrum by filling boost histogram with entries at
        # bin centers and weights equal to rescaled contents
        return Spectrum.from_filled(
            contents=self.get_contents(flow=True) * scale,
            errors=self.get_errors(flow=True) * scale,
            exposure=new_exposure,
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            zaxis=self.zaxis,
        )

    def accumulate(self, other):
        accumulated = super().__add__(other)
        accumulated = self.exposure + other.exposure
        return accumulated

    def __add__(self, other):
        if isinstance(other, Spectrum):
            scaled_other = other.scale_to_exposure(self.exposure)
            summed = super().__add__(other)
            summed.exposure = self.exposure
            return accumulated

        elif isinstance(other, Histogram):
            raise TypeError(
                "Cannot add Spectrum, which has exposure, and Histogram, which does not"
            )
        else:
            spec = super().__add__(other)
            spec.exposure = self.exposure
            return spec

    def __sub__(self, other):
        if isinstance(other, Spectrum):
            scaled_other = other.scale_to_exposure(self.exposure)
            subtracted = super().__sub__(scaled_other)
            subtracted.exposure = self.exposure
            return subtracted
        elif isinstance(other, Histogram):
            raise TypeError(
                "Cannot add Spectrum, which has exposure, and Histogram, which does not"
            )
        else:
            spec = super().__sub__(other)
            spec.exposure = self.exposure
            return spec

    def __mul__(self, other):
        if isinstance(other, Histogram):
            spec = super().__mul__(other)
            spec.exposure = self.exposure
            return spec
        else:
            spec = super().__mul__(other)
            spec.exposure = self.exposure * other
            return spec

    def __truediv__(self, other):
        if isinstance(other, Spectrum):
            scaled_other = other.scale_to_exposure(self.exposure)
            return super().__truediv__(scaled_other)

        elif isinstance(other, Histogram):
            raise TypeError(
                "Cannot divide Spectrum, which has exposure, and Histogram, which does not"
            )
        else:
            spec = super().__truediv__(other)
            spec.exposure = self.exposure / other
            return spec

    def __eq__(self, other):
        eq_exposure = self.exposure == other.exposure
        return eq_exposure & super().__eq__(other)

    def gather(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        contents = self.get_contents(True)
        squared_errors = self.get_errors(True) ** 2

        contents = comm.reduce(contents, MPI.SUM, root=self.root)
        squared_errors = comm.reduce(squared_errors, MPI.SUM, root=self.root)
        exposure = comm.reduce(self.exposure, MPI.SUM, root=self.root)

        if comm.Get_rank() == self.root:
            result = Spectrum.from_filled(
                contents=contents,
                errors=np.sqrt(squared_errors),
                exposure=exposure,
                xaxis=self.xaxis,
                yaxis=self.yaxis,
                zaxis=self.zaxis,
            )
        else:
            result = Spectrum(*self.axes)

        return result

    def saveto(self, filename_or_handle, group_name):
        from mpi4py import MPI

        if isinstance(filename_or_handle, str):
            filename_or_handle = h5py.File(filename_or_handle, "r+")

        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            spec = self.gather()
        else:
            spec = self

        if comm.Get_rank() == self.root:
            group = filename_or_handle.create_group(group_name)
            group.create_dataset(
                "contents", data=spec.get_contents(flow=True), compression="gzip"
            )
            group.create_dataset(
                "errors", data=spec.get_errors(flow=True), compression="gzip"
            )
            group.attrs["exposure"] = spec.exposure
            AxisFactory.saveto(spec.xaxis, group, "xaxis")
            AxisFactory.saveto(spec.yaxis, group, "yaxis")
            AxisFactory.saveto(spec.zaxis, group, "zaxis")

            return group

    @staticmethod
    def loadfrom(filename_or_handle, group_name, return_group=False):
        hist, group = Histogram.loadfrom(
            filename_or_handle, group_name, return_group=True
        )
        spec = Spectrum.from_filled(
            contents=hist.get_contents(True),
            errors=hist.get_errors(True),
            exposure=group.attrs["exposure"],
            xaxis=hist.xaxis,
            yaxis=hist.yaxis,
            zaxis=hist.zaxis,
        )
        if return_group:
            return spec, group
        else:
            return spec

    def azimov_data(self, exposure):
        return self.scale_to_exposure(exposure)

    def mock_data(self, exposure=-1, seed=0):
        exposure = exposure if exposure > 0 else self.exposure
        scale = exposure / self.exposure
        rng = np.random.default_rng(seed)
        orig = self.get_contents(flow=True)
        fluctuated = rng.poisson(orig * scale, orig.shape)
        errors = np.sqrt(fluctuated)
        return Spectrum.from_filled(
            contents=fluctuated,
            errors=errors,
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            zaxis=self.zaxis,
            exposure=exposure,
        )


def activeguard(factory=None):
    def dec(fun):
        def wrap(self, *args, **kwargs):
            if self.active:
                return fun(self, *args, **kwargs)
            else:
                if factory:
                    return factory()

        return wrap

    return dec


class ActiveObject:
    def __init__(self, root):
        from mpi4py import MPI

        self.root = root
        self.active = MPI.COMM_WORLD.Get_rank() == root


class Dataset(ActiveObject):
    def __init__(self, h5ds=None, root=0):
        super().__init__(root)
        self.h5ds = h5ds

    @property
    @activeguard(factory=dict)
    def attrs(self):
        return self.h5ds.attrs


class Handle:
    def __init__(self):
        pass

    @abstractmethod
    def create_group(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, key):
        pass


class ActiveHandle(ActiveObject):
    def __init__(self, h5obj=None, root=0):
        super().__init__(root)
        self.h5obj = h5obj

    @activeguard(factory=Dataset)
    def create_dataset(self, *args, **kwargs):
        return Dataset(h5ds=self.h5obj.create_dataset(*args, **kwargs), root=self.root)

    @activeguard(factory=Handle)
    def create_group(self, *args, **kwargs):
        return Group(h5group=self.h5obj.create_group(*args, **kwargs), root=self.root)

    @property
    @activeguard(factory=dict)
    def attrs(self):
        return self.h5obj.attrs

    @activeguard(factory=Dataset)
    def get(self, key):
        return self.h5obj[key]

    @activeguard(factory=list)
    def __iter__(self):
        yield self.h5obj.__iter__()


class Group(ActiveHandle):
    def __init__(self, h5group=None, root=0):
        super().__init__(h5obj=h5group, root=root)

    """
    @activeguard(factory=Dataset)
    def create_dataset(self, *args, **kwargs):
        return Dataset(h5ds=self.h5group.create_dataset(*args, **kwargs),
                       root=self.root)

    @activeguard()
    def create_group(self, *args, **kwargs):
        return Group(h5group=self.h5group.create_group(*args, **kwargs),
                     root=self.root)
    """


class File(ActiveHandle):
    def __init__(self, *args, root=0, **kwargs):
        super().__init__(h5obj=None, root=root)
        self.h5obj = self.open(*args, **kwargs)

    @activeguard()
    def open(self, *args, **kwargs):
        return h5py.File(*args, **kwargs)

    @activeguard()
    def close(self):
        self.h5obj.close()

    """
    @activeguard(factory=Group)
    def create_group(self, *args, **kwargs):
        return Group(h5group=self.h5file.create_group(*args, **kwargs),
                     root=self.root)

    @activeguard(factory=Dataset)
    def create_dataset(self, *args, **kwargs):
        return Dataset(h5ds=self.h5file.create_dataset(*args, **kwargs),
                       root=self.root)
    """

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
