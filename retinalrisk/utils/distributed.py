import os

import torch.distributed as dist


def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "17392"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()
