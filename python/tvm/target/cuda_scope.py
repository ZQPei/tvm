# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Global configuration/variable scope for cuda target"""
import tvm
from tvm._ffi import register_func as _register_func
from tvm.contrib import nvcc

from .. import nd as _nd


class CudaGlobalScope(object):
    current = None

    def __init__(self):
        self._old = CudaGlobalScope.current
        CudaGlobalScope.current = self

        self._old_cuda_target_arch = None
        self.cuda_target_arch = None


CUDA_GLOBAL_SCOPE = CudaGlobalScope()
CUDA_DEVICE_TYPE = _nd.device("cuda").device_type


@_register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    curr_cuda_target_arch = get_cuda_target_arch()
    # e.g., target arch could be [
    #   "-gencode", "arch=compute_52,code=sm_52",
    #   "-gencode", "arch=compute_70,code=sm_70"
    # ]
    compile_target = "fatbin" if isinstance(curr_cuda_target_arch, list) else "ptx"
    ptx = nvcc.compile_cuda(code, compile_target=compile_target, arch=curr_cuda_target_arch)
    return ptx


@_register_func("target.set_cuda_target_arch")
def set_cuda_target_arch(arch, gencode=True):
    """set target architecture of nvcc compiler

    Parameters
    ----------
    arch: str or list
        The argument of nvcc -arch. (e.g. "sm_51", "sm_62")
        it can also be a count of gencode arguments pass to nvcc command line,
        e.g., ["-gencode", "arch=compute_52,code=sm_52", "-gencode", "arch=compute_70,code=sm_70"]
    """
    if arch is None:
        cuda_arch = None
    elif isinstance(arch, str):
        if gencode:
            compute_version = arch.split('_')[1]
            cuda_arch = [
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}"
            ]
        else:
            cuda_arch = arch
    elif isinstance(arch, (list, tvm.ir.container.Array)):
        cuda_arch = list(arch)
        assert "-gencode" in cuda_arch
    else:
        raise TypeError(
            "arch is expected to be str or "
            + "a list or NoneType, but received "
            + "{}".format(type(arch)))

    CudaGlobalScope.current._old_cuda_target_arch = get_cuda_target_arch()
    CudaGlobalScope.current.cuda_target_arch = cuda_arch


@_register_func("target.get_cuda_target_arch")
def get_cuda_target_arch(target=None):
    """get target architecture of nvcc compiler"""
    if target is None:
        if isinstance(CudaGlobalScope.current.cuda_target_arch, tvm.ir.container.Array):
            return list(CudaGlobalScope.current.cuda_target_arch)
        return CudaGlobalScope.current.cuda_target_arch

    else:
        return tvm.target.Target(target=target).arch


@_register_func("target.reset_cuda_target_arch")
def reset_cuda_target_arch():
    """reset target architecture of nvcc compiler to None"""
    CudaGlobalScope.current.cuda_target_arch = None
