import numpy as np
from typing_extensions import Protocol
from typing import Union, Tuple, List
from mpi4py import MPI


class array:
    """ Wraps a numpy array for data-parallel operations
    """

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


class SupportsArray(Protocol):
    def __array__(self: "SupportsArray") -> float:
        pass


class histogram(array):
    """ Histogram events as a function of some event property
    with associated exposure
    """

    def __init__(
        self,
        x: Union[SupportsArray, None],
        bins: np.array,
        exposure: float = 1,
        contents: Union[np.array, array] = None,
        **kwargs,
    ):
        """[summary]

        Args:
            x (SupportsArray): Data to be histogrammed. Object must be a list,
                               or implement __array__
            bins (np.array): Array of bin edges
            exposure (float, optional): Exposure representing data being histogrammed.
                                        Defaults to 1.
            contents (array, optional): Used to construct a filled histogram.
                                        Defaults to None.
            **kwargs : forwarded to np.histogram
        """
        if contents is None:
            n, bins = np.histogram(x, bins=bins, **kwargs)
        else:
            n = contents
        super().__init__(n)
        self._bins = bins
        self.exposure = exposure

    @property
    def bins(self) -> np.array:
        return self._bins

    @property
    def contents(self) -> np.array:
        return self._data

    def array(self) -> array:
        return self.data

    def __getitem__(self, item) -> "histogram":
        """Slice this histogram with 'item'.
        It's possible that this returns a histogram with fewer dimensions than this"""

        sliced = self.data[item]
        if sliced.ndim == 1:
            return histogram(
                None,
                bins=None,  # what to do with bins? is a sparse histogram needed?
                exposure=self.exposure,
                contents=sliced,
            )
        elif sliced.ndim == 2:
            return histogram2d(
                None,
                None,
                bins=None,  # what to do with bins? is a sparse histogram needed?
                exposure=self.exposure,
                contents=sliced,
            )

    def scale_by_exposure(self, exposure: float) -> "histogram":
        """Return a new histogram that has been scaled by the input exposure.
        Scaling is done by multiplying the contents of this histogram by 
        the ratio of the input exposure and the exposure this histogram was created with

        Args:
            exposure (float): Exposure to scale this histogram with

        Returns:
            histogram : Rescaled histogram
        """
        return type(self)(
            *([None] * self.data.ndim),
            self._bins,
            exposure=exposure,
            contents=self.contents * exposure / self.exposure,
        )

    def _bins_consistent_with(self, other: np.array):
        if np.array(self.bins).shape != np.array(other.bins).shape:
            return False
        else:
            return (self.bins == other.bins).all()

    def __truediv__(self, other: Union["histogram", int, float]) -> "histogram":
        """If other is a histogram, do bin by bin division.
        Scale other histogram to this exposure before operating. 
        The resulting histogram has exposure of 1

        Since a1 / a2 + b1 / b2 =/= (a1 + b1) / (a2 + b2)
        both histograms in this operation will be reduced then divided.

        If other is numeric, divide all contents by other

        Args: 
            other (histogram | int | float)
        Returns:
            histogram : new histogram
        """
        if isinstance(other, histogram):
            assert self._bins_consistent_with(other)

            # reduce
            self = self.eval()
            other = other.eval()

            return type(self)(
                *([None] * self.data.ndim),
                bins=self.bins,
                exposure=self.exposure,
                contents=self.contents
                / other.scale_by_exposure(self.exposure).contents,
            )
        elif isinstance(other, float) | isinstance(other, int):
            return type(self)(
                *([None] * self.data.ndim),
                self.bins,
                exposure=1,
                contents=self.contents / other,
            )
        else:
            raise TypeError

    def __eq__(self, other: "histogram") -> bool:
        """Two histograms are equal if their contents are equal when scaled
        to the same exposure

        TODO: Does this need to compare the global histograms or local ones?
        """
        if isinstance(other, histogram):
            if not self._bins_consistent_with(other):
                return False
            else:
                return (
                    self.contents == other.contents * self.exposure / other.exposure
                ).all()
        else:
            return False

    def __mul__(self, other: Union["histogram", int, float]) -> "histogram":
        """If other is a histogram, do bin by bin multiplication.
        The resulting histogram has exposure equal to the produce of the two
        histograms.

        Since a1 / a2 + b1 / b2 =/= (a1 + b1) / (a2 + b2)
        both histograms in this operation will be reduced then divided.

        If other is numeric, divide all contents by other

        Args: 
            other (histogram | int | float)
        Returns:
            histogram : new histogram
        """
        if isinstance(other, histogram):
            assert self._bins_consistent_with(other)

            # reduce
            self = self.eval()
            other = other.eval()

            return type(self)(
                *([None] * self.data.ndim),
                bins=self.bins,
                exposure=self.exposure * other.exposure,
                contents=self.contents * other.contents,
            )
        elif isinstance(other, float) | isinstance(other, int):
            return type(self)(
                *([None] * self.data.ndim),
                self.bins,
                exposure=self.exposure,
                contents=self.contents * other,
            )
        else:
            raise TypeError

    def __add__(self, other: Union["histogram", int, float]) -> "histogram":
        """If other is a histogram, do bin-by-bin addition.
        Scale other histogram to this exposure before operating.
        Since addition is commutative, no reductions occur

        If other is numeric, add all bin contents by other
        Args:
            other (histogram | int | float) : histogram or numeric to add to this histogram
        Returns:
            histogram : new histogram
        """
        if isinstance(other, histogram):
            return type(self)(
                None,
                self.bins,
                exposure=self.exposure,
                contents=self.contents
                + other.scale_by_exposure(self.exposure).contents,
            )
        elif isinstance(other, float) | isinstance(other, int):
            return type(self)(
                *([None] * self.data.ndim),
                self.bins,
                exposure=self.exposure,
                contents=self.contents + other,
            )
        else:
            raise TypeError

    def __sub__(self, other: "histogram") -> "histogram":
        """If other is a histogram, do bin-by-bin subtraction.
        Scale other histogram to this exposure before operating.
        Since addition is commutative, no reductions occur.


        If other is numeric, subtract all bin contents by other
        Args:
            other (histogram | int | float) : histogram or numeric to add to this histogram
        Returns:
            histogram : new histogram
        """
        return self.__add__(other * -1)

    def eval(self, to: int = 0):
        """Do any necessary MPI communications
        Args:
            to (int) : rank to send data to for aggregation. Can also be a 'all'
                       to broadcast data to all ranks after aggregation
        Returns:
            histogram : result of the evaluation
        """
        return self

    def sum(self, *args, **kwargs):
        # reduce with MPI.SUM?
        return type(self)(
            *([None] * self.data.ndim),
            bins=self.bins,
            exposure=self.exposure,
            contents=self._data.sum(*args, **kwargs),
        )

    def __repr__(self) -> str:
        return self.contents.__repr__()

    def __len__(self):
        return len(self.bins)


class histogram2d(histogram):
    def __init__(
        self,
        x: SupportsArray,
        y: SupportsArray,
        bins: Tuple[np.array, np.array],
        exposure: float = 1,
        contents: Union[np.array, array] = None,
        **kwargs,
    ):
        """[summary]

            Args:
                x (SupportsArray): Data to be histogrammed.
                                   Object must have __array__ function
                bins (Tuple[np.array, np.array]): Tuple of bin edges
                exposure (float, optional): Exposure representing the data being histogrammed.
                                            Defaults to 1.
                contents (array, optional): Used to construct a filled histogram.
                                            Defaults to None.
                **kwargs : forwarded to np.histogram2d
            """
        if contents is None:
            try:
                n, binsx, binsy = np.histogram2d(
                    x.__array__(), y.__array__(), bins=bins, **kwargs
                )
            except AttributeError:
                n, binsx, binsy = np.histogram2d(x, y, bins=bins, **kwargs)
            bins = (binsx, binsy)
        else:
            n = contents
        super().__init__(None, bins, exposure=exposure, contents=n)
