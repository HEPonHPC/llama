import numpy as np
#from typing_extensions import Protocol
from typing import Union, Tuple, List
from mpi4py import MPI
import boost_histogram as bh
import pandana

class Array:
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


#class SupportsArray(Protocol):
#    def __array__(self: "SupportsArray") -> float:
#        pass

class Axis:
    def __init__(self, 
                 *, 
                 start=None,
                 stop=None,
                 n=None,
                 edges=None,
                 name='',
    ):
        self.name = name
        if start is not None and stop is not None and n is not None:
            self.baxis = bh.axis.Regular(n,
                                        start,
                                        stop,
                                        growth=False,
                                        circular=False,
                                        overflow=True,
                                        underflow=True)
        elif edges:
            self.baxis = bh.axis.Variable(edges,
                                         growth=False,
                                         circular=False,
                                         overflow=True,
                                         underflow=True)
        else:
            self.baxis = None
            

    def __call__(self):
        return self.baxis

    def from_boost(boost_axis, name=''):
        ret = Axis()
        ret.name = name
        ret.axis = boost_axis

class Histogram(Array):
    """ Histogram events as a function of some event property
    """

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
        if yaxis: self.ndim = self.ndim+1
        if zaxis: self.ndim = self.ndim+1

        self.bhist = bh.Histogram(self.get_axes(),
                                  storage=bh.storage.Weight())

        self.set_w2()

    def set_w2(self):
        self.errors = np.sqrt(self.bhist.variances(flow=True))

    def get_axes(self): 
        if self.ndim == 1:
            return self.xaxis()
        elif self.ndim == 2:
            return self.xaxis(), self.yaxis()
        else:
            return self.xaxis(), self.yaxis(), self.zaxis()

    def fill(self, data, weights=None):
        assert data.ndim == self.ndim
        if self.ndim == 1:
            self.bhist.fill(data, weight=weights)
        else:
            self.bhist.fill(*np.hsplit(data, data.shape[1]), weight=weights)
        self.set_w2()

    def _bins_consistent_with(self, other: np.array):
        if np.array(self.bins).shape != np.array(other.bins).shape:
            return False
        else:
            return (self.bins == other.bins).all()

    def get_contents(self, flow=False):
        return self.bhist.values(flow=flow)

    def get_errors(self, flow=False):
        if flow:
            return self.errors
        else:
            return self.errors[1:-1]
            
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
        if isinstance(other, histogram):
            assert self._bins_consistent_with(other)

            # reduce
            self = self.eval()
            other = other.eval()

            return type(self)(
                *([None] * self.data.ndim),
                bins=self.bins,
                contents=self.contents / other.contents,
            )
        elif isinstance(other, float) | isinstance(other, int):
            return type(self)(
                *([None] * self.data.ndim),
                self.bins,
                contents=self.contents / other,
            )
        else:
            raise TypeError

    def __eq__(self, other: "histogram") -> bool:
        """Two histograms are equal if their contents are equal

        TODO: Does this need to compare the global histograms or local ones?
        """
        if isinstance(other, histogram):
            if not self._bins_consistent_with(other):
                return False
            else:
                return (
                    self.contents == other.contents
                ).all()
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
        if isinstance(other, histogram):
            assert self._bins_consistent_with(other)

            # reduce
            self = self.eval()
            other = other.eval()

            return type(self)(
                *([None] * self.data.ndim),
                bins=self.bins,
                contents=self.contents * other.contents,
            )
        elif isinstance(other, float) | isinstance(other, int):
            return type(self)(
                *([None] * self.data.ndim),
                self.bins,
                contents=self.contents * other,
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
        if isinstance(other, histogram):
            return type(self)(
                None,
                self.bins,
                contents=self.contents + other.contents,
            )
        elif isinstance(other, float) | isinstance(other, int):
            return type(self)(
                *([None] * self.data.ndim),
                self.bins,
                contents=self.contents + other,
            )
        else:
            raise TypeError

    def __sub__(self, other: "histogram") -> "histogram":
        """If other is a histogram, do bin-by-bin subtraction.
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
            contents=self._data.sum(*args, **kwargs),
        )

    def __len__(self):
        return len(self.bins)

class Spectrum(Histogram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exposure = None

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

    def scale_to_exposure(self, new_exposure):
        scale = new_exposure / old_exposure
        # make new spectrum by filling boost histogram with entries at
        # bin centers and weights equal to rescaled contents
        snew = Spectrum(xaxis, yaxis, zaxis)
        
        
    def __add__(self, other):
        if isinstance(other, Spectrum):
            pass
        else:
            pass
            
