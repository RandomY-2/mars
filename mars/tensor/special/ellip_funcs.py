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

from ..utils import infer_dtype, implement_scipy
from .core import TensorSpecialUnaryOp, _register_special_op, TensorSpecialMultiOp


@_register_special_op
class TensorEllipk(TensorSpecialUnaryOp):
    _func_name = "ellipk"


@implement_scipy(spspecial.ellipk)
@infer_dtype(spspecial.ellipk)
def ellipk(m, **kwargs):
    op = TensorEllipk(**kwargs)
    return op(m)
