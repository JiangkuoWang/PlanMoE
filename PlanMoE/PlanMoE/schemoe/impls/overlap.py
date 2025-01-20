# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import schemoe_custom_kernel
from time import time
from torch.distributed import get_rank

from ..impls import communicate as C


class Compress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, compress_name, comm_name):
        input = schemoe_custom_kernel.compress_operation(input, compress_name, comm_name)
        return input

    @staticmethod
    def backward(ctx, grad):
        grad = schemoe_custom_kernel.decompress_operation(grad)
        return grad, None, None


class Decompress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, compress_name, comm_name):
        ctx.compress_name = compress_name
        ctx.comm_name = comm_name
        input = schemoe_custom_kernel.decompress_operation(input)
        return input

    @staticmethod
    def backward(ctx, grad):
        return schemoe_custom_kernel.compress_operation(grad, ctx.compress_name, ctx.comm_name), None, None


class Comm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return schemoe_custom_kernel.comm_operation(input)

    @staticmethod
    def backward(ctx, grad):
        return schemoe_custom_kernel.comm_operation(grad)


def a2a_ffn_overlap_forward(input, expert_fn, a2a_ffn_overlap_degree, use_2dh, group, compress_name, comm_name):
    split_dim = 1
    assert a2a_ffn_overlap_degree <= C.AllToAllStatus.max_num_split, "Excepting a2a_ffn_overlap_degree (%d) <= AllToAllStatus.max_num_split (%d)." % (
        a2a_ffn_overlap_degree, C.AllToAllStatus.max_num_split)
    assert input.shape[split_dim] % a2a_ffn_overlap_degree == 0, "Excepting input.shape[%d] (%d) be multiple of a2a_ffn_overlap_degree (%d)." % (
        split_dim, input.shape[split_dim], a2a_ffn_overlap_degree)
    C.AllToAllStatus.init(group, a2a_ffn_overlap_degree, split_dim)

    split_size = input.shape[split_dim] // a2a_ffn_overlap_degree
    input_split = list(input.split(split_size, dim=split_dim))
    schemoe_custom_kernel.clear_ptr_lst()

    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = input_split[i].contiguous()
        input_split[i] = Compress.apply(input_split[i], compress_name, comm_name)
        # for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Comm.apply(input_split[i])

    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Decompress.apply(input_split[i], compress_name, comm_name)
        input_split[i] = C.post_expert_permute(
            expert_fn(C.pre_expert_permute(input_split[i], group=group)), group=group
        )
        input_split[i] = Compress.apply(input_split[i], compress_name, comm_name)
        # for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Comm.apply(input_split[i])
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Decompress.apply(input_split[i], compress_name, comm_name)
    output = torch.cat(input_split, dim=split_dim).contiguous()
    return output
   