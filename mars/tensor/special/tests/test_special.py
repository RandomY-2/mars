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

import numpy as np
from scipy.special import (
    gammaln as scipy_gammaln,
    erf as scipy_erf,
    erfc as scipy_erfc,
    erfcx as scipy_erfcx,
    erfi as scipy_erfi,
    erfinv as scipy_erfinv,
    erfcinv as scipy_erfcinv,
    wofz as scipy_wofz,
    dawsn as scipy_dawsn,
    fresnel as scipy_fresnel,
    betainc as scipy_betainc,
)

from ....core import tile
from ... import tensor
from ..err_fresnel import (
    erf,
    TensorErf,
    erfc,
    TensorErfc,
    erfcx,
    TensorErfcx,
    erfi,
    TensorErfi,
    erfinv,
    TensorErfinv,
    erfcinv,
    TensorErfcinv,
    wofz,
    TensorWofz,
    dawsn,
    TensorDawsn,
    fresnel,
    TensorFresnel,
)
from ..gamma_funcs import (
    gammaln,
    TensorGammaln,
    betainc,
    TensorBetaInc,
)


def test_gammaln():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = gammaln(t)
    expect = scipy_gammaln(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorGammaln)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_elf():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erf(t)
    expect = scipy_erf(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErf)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfc():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erfc(t)
    expect = scipy_erfc(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErfc)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfcx():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erfcx(t)
    expect = scipy_erfcx(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErfcx)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfi():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erfi(t)
    expect = scipy_erfi(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErfi)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfinv():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erfinv(t)
    expect = scipy_erfinv(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErfinv)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfcinv():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erfcinv(t)
    expect = scipy_erfcinv(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErfcinv)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_wofz():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = wofz(t)
    expect = scipy_wofz(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorWofz)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_dawsn():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = dawsn(t)
    expect = scipy_dawsn(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorDawsn)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_fresnel():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = fresnel(t)
    expect = scipy_fresnel(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorFresnel)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_beta_inc():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)

    r = betainc(a, b, c)
    expect = scipy_betainc(raw1, raw2, raw3)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorBetaInc)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape

    betainc(a, b, c, out=a)
    expect = scipy_betainc(raw1, raw2, raw3)

    assert a.shape == raw1.shape
    assert a.dtype == expect.dtype

    b, tiled_a = tile(b, a)

    assert tiled_a.nsplits == b.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorBetaInc)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape
