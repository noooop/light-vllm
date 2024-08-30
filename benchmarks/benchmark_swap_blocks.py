import torch
import time

from vllm.utils import FlexibleArgumentParser
from vllm import _custom_ops as ops


def benchmark_swap_in_blocks(src_shape, num_blocks):
    dst = torch.randn(src_shape, dtype=torch.float16).cuda()
    src = torch.zeros_like(dst).cpu()

    block_mapping = [(i, i) for i in range(num_blocks)]
    blocks_to_swap = torch.tensor(block_mapping,
                                  device="cpu",
                                  dtype=torch.int64).view(-1, 2)

    num_iterations = 100
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        ops.swap_blocks(src, dst, blocks_to_swap)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time = total_time / num_iterations
    print(
        f"Avg. CPU->GPU time taken for swapping blocks: {average_time*1000} ms"
    )


def benchmark_swap_out_blocks(src_shape, num_blocks):
    src = torch.randn(src_shape, dtype=torch.float16).cuda()
    dst = torch.zeros_like(src).cpu()
    block_mapping = [(i, i) for i in range(num_blocks)]
    blocks_to_swap = torch.tensor(block_mapping,
                                  device="cpu",
                                  dtype=torch.int64).view(-1, 2)

    num_iterations = 100
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        ops.swap_blocks(src, dst, blocks_to_swap)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time = total_time / num_iterations
    print(
        f"Avg. GPU->CPU time taken for swapping blocks: {average_time*1000} ms"
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-blocks", type=int, default="1024")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=32)
    args = parser.parse_args()
    print(args)

    src_shape = (args.num_blocks, args.block_size, args.num_kv_heads,
                 args.head_size)

    benchmark_swap_in_blocks(src_shape, args.num_blocks)
    benchmark_swap_out_blocks(src_shape, args.num_blocks)

