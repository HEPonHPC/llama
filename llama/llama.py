import numpy as np

# from typing_extensions import Protocol
from typing import Union, Tuple, List
from mpi4py import MPI
import boost_histogram as bh
import pandana


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
        self.baxis = bh.axis.Variable(
            *args, metadata=label, **AxisFactory.defaults, **kwargs
        )


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
            axes = self.xaxis
        elif self.ndim == 2:
            axes = (self.xaxis, self.yaxis)
        else:
            axes = (self.xaxis, self.yaxis, self.zaxis)

        self.bhist = bh.Histogram(axes, storage=bh.storage.Weight())

        self.set_w2()

        self.root = 0
        self.proxy = HistProxy(self)

    def set_root(self, rank):
        self.root = rank

    def reduce(self):
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

    def fill(self, data, weights=None):
        assert data.ndim == self.ndim
        if self.ndim == 1:
            self.bhist.fill(data, weight=weights)
        else:
            self.bhist.fill(*np.hsplit(data, data.shape[1]), weight=weights)
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

            return (pad_with_flow(ax) for ax in self.bhist.axes)
        else:
            return self.bhist.axes.centers

    def get_contents(self, flow=False):
        return self.bhist.values(flow=flow)

    def set_contents(self, contents):
        self.bhist.reset()
        flow = contents.shape == self.bhist.counts(flow=True).shape
        self.bhist.fill(*self.get_bin_centers(flow), weight=contents)

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
            quotient_errors = np.sqrt(
                other.get_errors(flow=True) ** 2 + self.get_errors(flow=True) ** 2
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
                contents=self.get_contents(flow=True) * other,
                errors=self.get_errors(flow=True),
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
        if isinstance(other, histogram):
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
            summed_contents = other.get_contents(flow=True) * self.get_contents(
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
        return self.bhist.kind

    def values():
        return self.bhist.values()

    def variances():
        return self.bhist.variances()

    def counts():
        return self.bhist.counts()

    @property
    def axes(self):
        return self.bhist.axes


class Spectrum(Histogram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exposure = None

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
        super().fill(data, weights)
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

    def __mult__(self, other):
        if isinstance(other, Histogram):
            raise TypeError("Cannot multiply Spectra")

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
