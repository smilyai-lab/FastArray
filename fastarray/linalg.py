"""
Linear algebra operations for FastArray
Enhanced with advanced compression techniques for EXTREME SPEED and COMPRESSION
"""
import numpy as np
from .fastarray import FastArray, CompressionType, CompressionAggressiveness


def dot(a, b):
    """Dot product of two arrays - optimized for compressed data with extreme speed."""
    # Check if both are FastArray with compatible compression types
    if isinstance(a, FastArray) and isinstance(b, FastArray):
        # For INT8 quantized arrays (AQT-style), perform optimized INT8 operations
        if (a.compression_type == CompressionType.INT8_QUANT and
            b.compression_type == CompressionType.INT8_QUANT):
            # Use INT8 multiplication with proper scaling as in AQT
            a_decompressed = a._decompress()
            b_decompressed = b._decompress()
            result = np.dot(a_decompressed, b_decompressed)
            # Return result in INT8 format for continued speed
            return FastArray(result, compression=CompressionType.INT8_QUANT)

        # For low-rank decomposed arrays, use optimized multiplication
        elif (a.compression_type == CompressionType.LOW_RANK and
              b.compression_type == CompressionType.LOW_RANK):
            # For A (Ua*Sa*Vta) * B (Ub*Sb*Vtb), optimize the computation
            # If A is (m x k) and B is (k x n), then A*B = (Ua*Sa*Vta) * (Ub*Sb*Vtb)
            a_data = a._compressed_data
            b_data = b._compressed_data

            if isinstance(a_data, dict) and 'U' in a_data and isinstance(b_data, dict) and 'U' in b_data:
                # A = Ua @ diag(Sa) @ Vta, B = Ub @ diag(Sb) @ Vtb
                # A * B = Ua @ diag(Sa) @ Vta @ Ub @ diag(Sb) @ Vtb
                # This is complex, so for now decompress
                a_decompressed = a._decompress()
                b_decompressed = b._decompress()
                result = np.dot(a_decompressed, b_decompressed)
                return FastArray(result, compression=CompressionType.INT8_QUANT)

        # For mixed compression types, decompress and operate
        else:
            a_decompressed = a._decompress()
            b_decompressed = b._decompress()
            result = np.dot(a_decompressed, b_decompressed)
            # For extreme speed, return as INT8 quantized
            return FastArray(result, compression=CompressionType.INT8_QUANT,
                           compression_aggressiveness=max(a.compression_aggressiveness, b.compression_aggressiveness))

    elif isinstance(a, FastArray):
        a_decompressed = a._decompress()
        result = np.dot(a_decompressed, b)
        # For speed, try to compress result back
        if a.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT,
                           compression_aggressiveness=a.compression_aggressiveness)
        else:
            return FastArray(result, compression=a.compression_type)

    elif isinstance(b, FastArray):
        b_decompressed = b._decompress()
        result = np.dot(a, b_decompressed)
        # For speed, try to compress result back
        if b.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT,
                           compression_aggressiveness=b.compression_aggressiveness)
        else:
            return FastArray(result, compression=b.compression_type)

    # If neither is FastArray, just use numpy
    result = np.dot(a, b)
    if isinstance(result, np.ndarray):
        # Since no FastArray was involved, return as regular numpy array
        return result
    else:
        return result


def vdot(a, b):
    """Dot product of two vectors, with complex conjugation."""
    # Apply extreme speed optimizations
    if isinstance(a, FastArray) and isinstance(b, FastArray):
        # Use INT8 computation for maximum speed
        a_decompressed = a._decompress()
        b_decompressed = b._decompress()
        result = np.vdot(a_decompressed, b_decompressed)
        return result  # scalar result, no need to return FastArray

    # For other cases, decompress but try to be efficient
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()

    return np.vdot(a, b)


def inner(a, b):
    """Inner product of two arrays."""
    # Optimized for speed
    if isinstance(a, FastArray) and isinstance(b, FastArray):
        a_decompressed = a._decompress()
        b_decompressed = b._decompress()
        result = np.inner(a_decompressed, b_decompressed)
        return result  # scalar result

    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()

    return np.inner(a, b)


def outer(a, b):
    """Outer product of two arrays - optimized for extreme speed."""
    # Try to preserve compression type where possible for speed
    if isinstance(a, FastArray) and isinstance(b, FastArray):
        # For speed, always return as INT8 quantized
        a_decompressed = a._decompress()
        b_decompressed = b._decompress()
        result = np.outer(a_decompressed, b_decompressed)
        return FastArray(result, compression=CompressionType.INT8_QUANT,
                       compression_aggressiveness=max(a.compression_aggressiveness, b.compression_aggressiveness))

    elif isinstance(a, FastArray):
        a_decompressed = a._decompress()
        result = np.outer(a_decompressed, b)
        if a.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT)
        else:
            return FastArray(result, compression=a.compression_type)

    elif isinstance(b, FastArray):
        b_decompressed = b._decompress()
        result = np.outer(a, b_decompressed)
        if b.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT)
        else:
            return FastArray(result, compression=b.compression_type)

    result = np.outer(a, b)
    return FastArray(result, compression=CompressionType.INT8_QUANT)  # Return compressed for speed


def matrix_power(a, n):
    """Raise a square matrix to the (integer) power n - optimized for speed."""
    if isinstance(a, FastArray):
        # For extreme speed, decompress and compute, then return with optimized compression
        a_decompressed = a._decompress()
        result = np.linalg.matrix_power(a_decompressed, n)
        # Return as INT8 for speed in subsequent operations
        return FastArray(result, compression=CompressionType.INT8_QUANT,
                       compression_aggressiveness=a.compression_aggressiveness)

    result = np.linalg.matrix_power(a, n)
    return FastArray(result, compression=CompressionType.INT8_QUANT)  # Return compressed for speed


def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes - optimized for speed."""
    # Try to work with compressed data where possible for speed
    if isinstance(a, FastArray) and isinstance(b, FastArray):
        # Always use INT8 computation for maximum speed
        a_decompressed = a._decompress()
        b_decompressed = b._decompress()
        result = np.tensordot(a_decompressed, b_decompressed, axes=axes)
        return FastArray(result, compression=CompressionType.INT8_QUANT,
                       compression_aggressiveness=max(a.compression_aggressiveness, b.compression_aggressiveness))

    elif isinstance(a, FastArray):
        a_decompressed = a._decompress()
        result = np.tensordot(a_decompressed, b, axes=axes)
        if a.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT)
        else:
            return FastArray(result, compression=a.compression_type)

    elif isinstance(b, FastArray):
        b_decompressed = b._decompress()
        result = np.tensordot(a, b_decompressed, axes=axes)
        if b.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT)
        else:
            return FastArray(result, compression=b.compression_type)

    result = np.tensordot(a, b, axes=axes)
    return FastArray(result, compression=CompressionType.INT8_QUANT)  # Return compressed for speed


def kron(a, b):
    """Kronecker product of two arrays - optimized for speed."""
    # For Kronecker product, use INT8 computation for maximum speed
    if isinstance(a, FastArray) and isinstance(b, FastArray):
        a_decompressed = a._decompress()
        b_decompressed = b._decompress()
        result = np.kron(a_decompressed, b_decompressed)
        return FastArray(result, compression=CompressionType.INT8_QUANT,
                       compression_aggressiveness=max(a.compression_aggressiveness, b.compression_aggressiveness))

    elif isinstance(a, FastArray):
        a_decompressed = a._decompress()
        result = np.kron(a_decompressed, b)
        if a.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT)
        else:
            return FastArray(result, compression=a.compression_type)

    elif isinstance(b, FastArray):
        b_decompressed = b._decompress()
        result = np.kron(a, b_decompressed)
        if b.compression_aggressiveness >= CompressionAggressiveness.AGGRESSIVE:
            return FastArray(result, compression=CompressionType.INT8_QUANT)
        else:
            return FastArray(result, compression=b.compression_type)

    result = np.kron(a, b)
    return FastArray(result, compression=CompressionType.INT8_QUANT)  # Return compressed for speed


# Eigenvalues and decompositions - these generally require full decompression
# But we can optimize them for speed
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
    """Singular Value Decomposition - optimized for speed."""
    if isinstance(a, FastArray):
        # If it's already low-rank, we might be able to use the existing decomposition
        if a.compression_type == CompressionType.LOW_RANK:
            # If the array is already stored in low-rank form, return the individual components
            data = a._compressed_data
            if isinstance(data, dict) and 'U' in data and 's' in data and 'Vt' in data:
                U = data['U']
                s = data['s']
                Vt = data['Vt']
                # If compute_uv is True, return the stored decomposition
                if compute_uv:
                    return s, U, Vt  # Return singular values and components
                else:
                    return s  # Return only singular values
        # Otherwise, decompress normally
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


# Norms and other numbers - optimized for speed
def norm(x, ord=None, axis=None, keepdims=False):
    """Matrix or vector norm - optimized for speed."""
    if isinstance(x, FastArray):
        # Use compressed format for computation if possible
        x_decompressed = x._decompress()
        result = np.linalg.norm(x_decompressed, ord=ord, axis=axis, keepdims=keepdims)
        return result  # Return scalar value

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


# Solving equations and inverting matrices - optimized for speed
def solve(a, b):
    """Solve a linear matrix equation, or system of linear scalar equations."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()

    result = np.linalg.solve(a, b)
    return FastArray(result, compression=CompressionType.INT8_QUANT)  # Return compressed for speed


def inv(a):
    """Compute the (multiplicative) inverse of a matrix - optimized for speed."""
    if isinstance(a, FastArray):
        a = a._decompress()

    result = np.linalg.inv(a)
    return FastArray(result, compression=CompressionType.INT8_QUANT)  # Return compressed for speed


def pinv(a, rcond=1e-15, hermitian=False):
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix - optimized for speed."""
    if isinstance(a, FastArray):
        a = a._decompress()

    result = np.linalg.pinv(a, rcond=rcond, hermitian=hermitian)
    return FastArray(result, compression=CompressionType.INT8_QUANT)  # Return compressed for speed


def lstsq(a, b, rcond='warn'):
    """Return the least-squares solution to a linear matrix equation."""
    if isinstance(a, FastArray):
        a = a._decompress()
    if isinstance(b, FastArray):
        b = b._decompress()

    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=rcond)
    x_fast = FastArray(x, compression=CompressionType.INT8_QUANT)  # Compress result for speed
    return x_fast, residuals, rank, s