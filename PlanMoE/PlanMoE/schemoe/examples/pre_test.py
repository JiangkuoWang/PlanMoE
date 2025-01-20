#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, Tensor
import argparse
import schemoe_custom_kernel
import torch.distributed as dist
import math
from contextlib import nullcontext
from typing import Any
import time


def decorate_trace_handler(args, rank):
    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        if rank == 0:
            prof.export_chrome_trace(
                "./batch_size"
                + str(args.batch_size)
                + "#num_tokens"
                + str(args.num_tokens)
                + "#model_dim"
                + str(args.model_dim)
                + "#hidden_size"
                + str(args.hidden_size)
                + "#num_local_experts"
                + str(args.num_local_experts)
                + "#capacity_factor"
                + str(args.capacity_factor)
                + "#a2a_ffn_overlap_degree"
                + str(args.a2a_ffn_overlap_degree)
                + "#step_num"
                + str(prof.step_num)
                + ".json"
            )

    return trace_handler


parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_tokens", type=int, default=512)
parser.add_argument("--model_dim", type=int, default=2048)
parser.add_argument("--hidden_size", type=int, default=2048)
parser.add_argument("--num_local_experts", type=int, default=2)
parser.add_argument("--dtype", type=str, default="float32")
parser.add_argument("--fp32_gate", default=False, action="store_true")
parser.add_argument("--top", type=int, default=2)
parser.add_argument("--a2a_ffn_overlap_degree", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=25)
parser.add_argument("--capacity_factor", type=float, default=1.0)
parser.add_argument("--parallel_type", type=str, default="auto")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--use_2dh", default=False, action="store_true")
parser.add_argument("--record_shapes", default=False, action="store_true")
parser.add_argument("--with_stack", default=False, action="store_true")
parser.add_argument("--log", type=str, default="test.log")
parser.add_argument("--encode", type=str, default="no")


args = parser.parse_args()

dist.init_process_group("nccl")

dist_rank, dist_world_size = dist.get_rank(), dist.get_world_size()

args.local_rank = os.environ.get("LOCAL_RANK", 0)


def dist_print(*args):
    if dist_rank == 0:
        print(*args)


device = torch.device("cuda:%s" % args.local_rank)
torch.cuda.set_device(device)

torch.set_printoptions(sci_mode=False)

if args.dtype == "float32":
    torch.set_default_dtype(torch.float32)
elif args.dtype == "float64":
    torch.set_default_dtype(torch.float64)
elif args.dtype == "float16":
    torch.set_default_dtype(torch.float16)
elif args.dtype == "bfloat16":
    torch.set_default_dtype(torch.bfloat16)
else:
    raise Exception("Unrecognized data type specified: %s" % args.dtype)

from schemoe.impls import communicate as C

torch.manual_seed(0)


def single_case(
    batch_size,
    num_tokens,
    model_dim,
    hidden_size,
    num_local_experts,
    top_value,
    a2a_ffn_overlap_degree,
    capacity_factor,
):
    fc1_weight = torch.randn(
        num_local_experts,
        model_dim,
        hidden_size,
        dtype=torch.get_default_dtype(),
        device=device,
    )
    fc2_weight = torch.randn(
        num_local_experts,
        hidden_size,
        model_dim,
        dtype=torch.get_default_dtype(),
        device=device,
    )

    def zc(x, y):
        return (x + y - 1) // y * y

    expert_num = num_local_experts * dist_world_size
    x = torch.tensor(
        torch.randn(
            [
                expert_num,
                zc(
                    int(top_value * math.ceil(batch_size * num_tokens / expert_num) * capacity_factor),
                    a2a_ffn_overlap_degree if args.encode != "zfp" else a2a_ffn_overlap_degree * 4,
                ),
                model_dim,
            ],
            dtype=torch.float32,
            device="cpu",
        )
        .detach()
        .numpy(),
        dtype=torch.get_default_dtype(),
        requires_grad=False,
        device=device,
    )
    lst = []

    tuples = (
        dist_world_size,
        args.dtype,
        model_dim,
        hidden_size,
        batch_size * num_tokens,
        num_local_experts,
        top_value,
        a2a_ffn_overlap_degree,
        capacity_factor,
        device,
    )
    dist_print(
        "[Benchmark] world_size = %s, dtype = %s, model_dim = %s, hidden_size = %s, samples = %s, num_local_experts = %s, topK = %s, a2a_ffn_overlap_degree = %s, capacity_factor = `%s`, device = `%s`"
        % tuples
    )

    if dist_rank == 0:
        with open(args.log, "a+") as f:
            f.write(str(batch_size) + "," + str(num_tokens) + "," + str(model_dim) + "," + str(hidden_size) + "," + "Naive" + "," + str(capacity_factor) + "," + str(a2a_ffn_overlap_degree) + ",")
    C.AllToAllStatus.init(dist.group.WORLD, a2a_ffn_overlap_degree, 1)
    with torch.no_grad():
        for _ in range(args.num_steps):
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            schemoe_custom_kernel.clear_ptr_lst()
            # cuda_start.record()
            input = x.clone()
            # y = simple_all_to_all(y)
            # y = AllToAll2DAsync.apply(y)
            # cuda_end.record()
            # torch.cuda.synchronize()
            # if dist_rank == 0:
            #    lst.append(cuda_start.elapsed_time(cuda_end))
            cuda_start.record()
            split_size = input.shape[1] // a2a_ffn_overlap_degree
            input_split = list(input.split(split_size, dim=1))
            for i in range(a2a_ffn_overlap_degree):
                input_split[i] = input_split[i].contiguous()

                # input_size = input_split[i].size()
                # input_split[i] = input_split[i].view((-1, input_size[-1]))
                # cuda_start.record()
                input_split[i] = schemoe_custom_kernel.compress_operation(input_split[i], args.encode, "naive")
                # print(input_split[i].storage())
                input_split[i] = schemoe_custom_kernel.comm_operation(input_split[i])
            for i in range(a2a_ffn_overlap_degree):
                input_split[i] = schemoe_custom_kernel.decompress_operation(input_split[i])
                # input_split[i] = input_split[i].view(input_size)
                # input_split[i] = torch.matmul(input_split[i], fc1_weight)
                # input_split[i] = torch.nn.functional.relu(input_split[i])
                # input_split[i] = torch.matmul(input_split[i], fc2_weight)
                # input_split[i] = input_split[i].view((-1, input_size[-1]))
                input_split[i] = schemoe_custom_kernel.compress_operation(input_split[i], args.encode, "naive")
                input_split[i] = schemoe_custom_kernel.comm_operation(input_split[i])
            for i in range(a2a_ffn_overlap_degree):
                input_split[i] = schemoe_custom_kernel.decompress_operation(input_split[i])
                # input_split[i] = input_split[i].view(input_size)
            output = torch.cat(input_split, dim=1)
            print(output - input)
            cuda_end.record()
            torch.cuda.synchronize()
            if dist_rank == 0:
                lst.append(cuda_start.elapsed_time(cuda_end))
            torch.distributed.barrier()
            if dist_rank == 0:
                print("step:", _)
    if dist_rank == 0:
        with open(args.log, "a+") as f:
            f.write(str(lst[5:]) + "\n")


# 512, 1024, 2048, 4096, 8192

for batch_size in [
    8,
]:
    for num_tokens in [
        2048,
    ]:
        for model_dim in [
            1024,
        ]:
            for hidden_size in [
                1024,
            ]:
                for num_local_experts in [
                    1,
                ]:
                    for top_value in [
                        2,
                    ]:
                        for capacity_factor in [
                            1.2,
                        ]:
                            single_case(
                                batch_size,
                                num_tokens,
                                model_dim,
                                hidden_size,
                                num_local_experts,
                                top_value,
                                args.a2a_ffn_overlap_degree,
                                capacity_factor,
                            )
