"""
Linear algebra operations for FastArray
"""
import numpy as np
from .fastarray import FastArray


def dot(a, b):
    """Dot product of two arrays."""
    # Ensure both inputs are decompressed for the operation
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    result = np.dot(a, b)
    # Return as FastArray if result is an array, otherwise return scalar
    if isinstance(result, np.ndarray):
        return FastArray(result)
    else:
        return result


def vdot(a, b):
    """Dot product of two vectors, with complex conjugation."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    result = np.vdot(a, b)
    if isinstance(result, np.ndarray):
        return FastArray(result)
    else:
        return result


def inner(a, b):
    """Inner product of two arrays."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    result = np.inner(a, b)
    if isinstance(result, np.ndarray):
        return FastArray(result)
    else:
        return result


def outer(a, b):
    """Outer product of two arrays."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    result = np.outer(a, b)
    return FastArray(result)


def matrix_power(a, n):
    """Raise a square matrix to the (integer) power n."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    result = np.linalg.matrix_power(a, n)
    return FastArray(result)


def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    result = np.tensordot(a, b, axes=axes)
    return FastArray(result)


def kron(a, b):
    """Kronecker product of two arrays."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    result = np.kron(a, b)
    return FastArray(result)


# Eigenvalues and decompositions
def eig(a):
    """Compute the eigenvalues and right eigenvectors of a square array."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    return np.linalg.eig(a)


def eigh(a, UPLO='L'):
    """Return the eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    return np.linalg.eigh(a)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """Singular Value Decomposition."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)


def qr(a, mode='reduced'):
    """Compute the qr factorization of a matrix."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    return np.linalg.qr(a, mode=mode)


def cholesky(a):
    """Cholesky decomposition."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    return np.linalg.cholesky(a)


# Norms and other numbers
def norm(x, ord=None, axis=None, keepdims=False):
    """Matrix or vector norm."""
    if isinstance(x, FastArray):
        x = x._decompress()
    
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def cond(x, p=None):
    """Compute the condition number of a matrix."""
    if isinstance(x, FastArray):
        x = x._decompress()
    
    return np.linalg.cond(x, p=p)


def det(a):
    """Compute the determinant of an array."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    return np.linalg.det(a)


def matrix_rank(M, tol=None, hermitian=False):
    """Return matrix rank of array using SVD method."""
    if isinstance(M, FastArray):
        M = M._decompress()
    
    return np.linalg.matrix_rank(M, tol=tol, hermitian=hermitian)


def slogdet(a):
    """Compute the sign and (natural) logarithm of the determinant of an array."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    return np.linalg.slogdet(a)


# Solving equations and inverting matrices
def solve(a, b):
    """Solve a linear matrix equation, or system of linear scalar equations."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    result = np.linalg.solve(a, b)
    if isinstance(result, np.ndarray):
        return FastArray(result)
    else:
        return result


def inv(a):
    """Compute the (multiplicative) inverse of a matrix."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    result = np.linalg.inv(a)
    return FastArray(result)


def pinv(a, rcond=1e-15, hermitian=False):
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix."""
    if isinstance(a, FastArray):
        a = a._decompress()
    
    result = np.linalg.pinv(a, rcond=rcond, hermitian=hermitian)
    return FastArray(result)


def lstsq(a, b, rcond='warn'):
    """Return the least-squares solution to a linear matrix equation."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()
    
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=rcond)
    x_fast = FastArray(x)
    return x_fast, residuals, rank, s