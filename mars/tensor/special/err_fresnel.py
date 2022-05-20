# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import scipy.special as spspecial

from ..arithmetic.utils import arithmetic_operand
from ..utils import infer_dtype, implement_scipy
from .core import TensorSpecialUnaryOp, _register_special_op


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErf(TensorSpecialUnaryOp):
    _func_name = "erf"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfc(TensorSpecialUnaryOp):
    _func_name = "erfc"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfcx(TensorSpecialUnaryOp):
    _func_name = "erfcx"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfi(TensorSpecialUnaryOp):
    _func_name = "erfi"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfinv(TensorSpecialUnaryOp):
    _func_name = "erfinv"


@implement_scipy(spspecial.erf)
@infer_dtype(spspecial.erf)
def erf(x, out=None, where=None, **kwargs):
    """
    Returns the error function of complex argument.

    It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    res : Tensor
        The values of the error function at the given points `x`.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    Notes
    -----
    The cumulative of the unit normal distribution is given by
    ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover,
        1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm
    .. [3] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.tensor import special
    >>> import matplotlib.pyplot as plt
    >>> x = mt.linspace(-3, 3)
    >>> plt.plot(x, special.erf(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erf(x)$')
    >>> plt.show()
    """
    op = TensorErf(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfc)
@infer_dtype(spspecial.erfc)
def erfc(x, out=None, where=None, **kwargs):
    """
    Returns the complementary error function

    It is defined as ``1 - erf(x)``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    out: Tensor or None
        Optional output tensor for the function results.

    Returns
    -------
    res : Scalar or Tensor
        Values of the complementary error function.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation. http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.tensor import special
    >>> import matplotlib.pyplot as plt
    >>> x = mt.linspace(-3, 3)
    >>> plt.plot(x, special.erfc(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfc(x)$')
    >>> plt.show()
    """
    op = TensorErfc(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfcx)
@infer_dtype(spspecial.erfcx)
def erfcx(x, out=None, where=None, **kwargs):
    """
    Scales complementary error function

    It is defined as ``exp(x**2) * erfc(x)``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    out: Tensor or None
        Optional output tensor for the function results.

    Returns
    -------
    res : Scalar or Tensor
        Values of the scaled complementary error function.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation. http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.tensor import special
    >>> import matplotlib.pyplot as plt
    >>> x = mt.linspace(-3, 3)
    >>> plt.plot(x, special.erfcx(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfcx(x)$')
    >>> plt.show()
    """
    op = TensorErfcx(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfi)
@infer_dtype(spspecial.erfi)
def erfi(x, out=None, where=None, **kwargs):
    """
    Imaginary error function

    It is defined as ``-i erf(i, x)``.

    Parameters
    ----------
    x : Tensor
        Input tensor, real or complex valued argument.

    out: Tensor or None
        Optional output tensor for the function results.

    Returns
    -------
    res : Scalar or Tensor
        Values of the imaginary error function.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation. http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.tensor import special
    >>> import matplotlib.pyplot as plt
    >>> x = mt.linspace(-3, 3)
    >>> plt.plot(x, special.erfi(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfi(x)$')
    >>> plt.show()
    """
    op = TensorErfi(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfinv)
@infer_dtype(spspecial.erfinv)
def erfinv(x, out=None, where=None, **kwargs):
    """
    Inverse of the error function

    It is defined as ``-i erf(i, x)``.

    Parameters
    ----------
    x : Tensor
        Argument at which to evaluate. Domain: [-1, 1].

    Returns
    -------
    res : Tensor
        The inverse of erf of x, element-wise.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.tensor import special
    >>> special.erfinv(0.5)
    >>> x = mt.linspace(-1.0, 1.0, num=10)
    >>> special.erfinv(x)
    """
    op = TensorErfinv(**kwargs)
    return op(x, out=out, where=where)
