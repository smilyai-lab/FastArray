"""
Module to maintain NumPy API compatibility
This module exposes all standard NumPy functions that should work with FastArray
"""
import numpy as np

# Import all NumPy functions to maintain compatibility
# This allows all NumPy operations to work seamlessly with FastArray

# Mathematical functions
abs = np.abs
absolute = np.absolute
add = np.add
angle = np.angle
arccos = np.arccos
arccosh = np.arccosh
arcsin = np.arcsin
arcsinh = np.arcsinh
arctan = np.arctan
arctan2 = np.arctan2
arctanh = np.arctanh
around = np.around
ceil = np.ceil
conj = np.conj
conjugate = np.conjugate
copysign = np.copysign
cos = np.cos
cosh = np.cosh
deg2rad = np.deg2rad
degrees = np.degrees
divide = np.divide
divmod = np.divmod
equal = np.equal
exp = np.exp
exp2 = np.exp2
expm1 = np.expm1
fabs = np.fabs
floor = np.floor
floor_divide = np.floor_divide
fmax = np.fmax
fmin = np.fmin
fmod = np.fmod
frexp = np.frexp
greater = np.greater
greater_equal = np.greater_equal
heaviside = np.heaviside
hypot = np.hypot
imag = np.imag
invert = np.invert
isfinite = np.isfinite
isinf = np.isinf
isnan = np.isnan
isnat = np.isnat
ldexp = np.ldexp
left_shift = np.left_shift
less = np.less
less_equal = np.less_equal
log = np.log
log10 = np.log10
log1p = np.log1p
log2 = np.log2
logaddexp = np.logaddexp
logaddexp2 = np.logaddexp2
logical_and = np.logical_and
logical_not = np.logical_not
logical_or = np.logical_or
logical_xor = np.logical_xor
matmul = np.matmul
maximum = np.maximum
minimum = np.minimum
mod = np.mod
modf = np.modf
multiply = np.multiply
negative = np.negative
not_equal = np.not_equal
positive = np.positive
power = np.power
rad2deg = np.rad2deg
radians = np.radians
real = np.real
reciprocal = np.reciprocal
remainder = np.remainder
right_shift = np.right_shift
rint = np.rint
sign = np.sign
signbit = np.signbit
sin = np.sin
sinh = np.sinh
sqrt = np.sqrt
square = np.square
subtract = np.subtract
tan = np.tan
tanh = np.tanh
true_divide = np.true_divide
trunc = np.trunc

# Other functions
arange = np.arange
array = np.array
asarray = np.asarray
asanyarray = np.asanyarray
ascontiguousarray = np.ascontiguousarray
asfortranarray = np.asfortranarray
copy = np.copy
empty = np.empty
empty_like = np.empty_like
eye = np.eye
frombuffer = np.frombuffer
fromfile = np.fromfile
fromfunction = np.fromfunction
fromiter = np.fromiter
full = np.full  # Note: fromstring was removed in NumPy 2.0, use frombuffer
full_like = np.full_like
identity = np.identity
linspace = np.linspace
logspace = np.logspace
meshgrid = np.meshgrid
ones = np.ones
ones_like = np.ones_like
zeros = np.zeros
zeros_like = np.zeros_like
tile = np.tile
repeat = np.repeat

# Linear algebra
from numpy import dot, inner, outer, matmul, tensordot, einsum
from numpy import trace, diag, triu, tril
from numpy import vdot, correlate, convolve

# Mathematical functions (reductions)
from numpy import sum, prod, mean, std, var, min, max, argmin, argmax
from numpy import median, average, percentile, quantile
from numpy import all, any, cumsum, cumprod
from numpy import nansum, nanprod, nanmean, nanvar, nanstd, nanmin, nanmax

# Array manipulation
from numpy import concatenate, stack, hstack, vstack, dstack, column_stack
from numpy import split, array_split, hsplit, vsplit, dsplit
from numpy import shape, reshape, ravel, diagonal, transpose
from numpy import swapaxes, moveaxis, squeeze, expand_dims, atleast_1d, atleast_2d, atleast_3d
from numpy import where, nonzero, argwhere, flatnonzero
from numpy import broadcast_arrays, broadcast_to

# Logic functions
from numpy import allclose, isclose, array_equal, array_equiv
from numpy import greater, greater_equal, less, less_equal, equal, not_equal

# Sorting, searching, counting
from numpy import argmax, argmin, argsort, sort
from numpy import maximum, minimum, fmax, fmin
from numpy import unique, in1d, intersect1d, setxor1d, union1d, setdiff1d
from numpy import count_nonzero, searchsorted, resize, nonzero, argwhere

# Statistics
from numpy import corrcoef, cov, histogram, histogram2d, histogramdd, bincount, digitize

# Random module
import numpy.random as random