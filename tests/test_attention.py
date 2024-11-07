import torch
import pytest
from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmetricAttention
from genmo.mochi_preview.dit.joint_model.rope_mixed import compute_mixed_rotation, create_position_matrix
import os
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.core.long_ctx_attention.ring.ring_flash_attn import (
    xdit_ring_flash_attn_func,
)
import genmo.mochi_preview.dit.joint_model.context_parallel as cp
from genmo.mochi_preview.dit.joint_model import get_usp_config
def init_dist(backend="nccl"):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(
        f"Initializing distributed environment with rank {rank}, world size {world_size}, local rank {local_rank}"
    )

    torch.cuda.set_device(local_rank)
    init_distributed_environment(rank=rank, world_size=world_size)
    # dist.init_process_group(backend=backend)
       # construct a hybrid sequence parallel config (ulysses=2, ring = world_size // 2)

    ulysses_degree, ring_degree = get_usp_config()
    if ulysses_degree is None and ring_degree is None:
        print(f"No usp config, use default config: ulysses_degree={world_size}, ring_degree=1")
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=1,
            ulysses_degree=world_size,
        )
    else:
        if ulysses_degree is None:
            ulysses_degree = world_size // ring_degree
        if ring_degree is None:
            ring_degree = world_size // ulysses_degree
        print(f"Use usp config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}")
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=ring_degree,
            ulysses_degree=ulysses_degree,
        )

    # activate the cp
    pg = torch.distributed.group.WORLD
    cp.set_cp_group(pg, list(range(world_size)), local_rank)
    assert cp.get_cp_rank_size() == (rank, world_size)
    return rank, world_size

def test_forward_xdit_matches_forward():
    # Initialize model parameters
    rank, world_size = init_dist()
    dim_x = 3072
    dim_y = 1536
    num_heads = 24
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    
    batch_size = 1
    seq_len_x = 1592
    seq_len_y = 256

    # Create model instance
    model = AsymmetricAttention(
        dim_x=dim_x,
        dim_y=dim_y,
        num_heads=num_heads,
        qkv_bias=False,
        qk_norm=True,
        update_y=True,
        attention_mode="sdpa",
        device=device,
    ).to(dtype)

    # Broadcast model parameters from rank 0 to ensure consistency
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)

    # Create input tensors
    x = torch.randn(batch_size, seq_len_x, dim_x, device=device, dtype=dtype)
    y = torch.randn(batch_size, seq_len_y, dim_y, device=device, dtype=dtype)
    scale_x = torch.randn(batch_size, dim_x, device=device, dtype=dtype)
    scale_y = torch.randn(batch_size, dim_y, device=device, dtype=dtype)

    # Create position encodings
    T, pH, pW = 2, 30, 53  # Example values, adjust as needed

    # Create position array and compute rotations
    N = x.size(1)  # Get actual sequence length

    # T: 2, pH: 30, pW: 53, device: cuda:0, dtype: torch.bfloat16, target_area: 36864 
    pos = create_position_matrix(T, pH=pH, pW=pW, device=device, dtype=dtype)  # (N, 3)

    # Initialize pos_frequencies with correct size
    pos_frequencies = torch.randn(3, num_heads, dim_x // num_heads // 2, device=device, dtype=dtype)

    torch.distributed.broadcast(pos_frequencies, src=0)

    # Compute rotations for actual sequence length
    # rope_cos torch.Size([1592, 24, 64]) 
    rope_cos, rope_sin = compute_mixed_rotation(
        freqs=pos_frequencies, 
        pos=pos[:N]  # Only use positions up to sequence length
    )  # Each are (N, num_heads, dim // 2)

    valid_y_seq = 9
    assert valid_y_seq <= y.size(1), f"valid_y_seq {valid_y_seq} <= y.size(1) {y.size(1)}"

    # Run both forward passes
    with torch.no_grad():

        torch.distributed.broadcast(x, src=0)
        torch.distributed.broadcast(y, src=0)
        torch.distributed.broadcast(scale_x, src=0)
        torch.distributed.broadcast(scale_y, src=0)
        
        torch.distributed.broadcast(rope_cos, src=0)
        torch.distributed.broadcast(rope_sin, src=0)

        total_len = seq_len_x + valid_y_seq
        valid_token_indices = torch.arange(total_len, device=device)
        cu_seqlens = torch.tensor([0, total_len], device=device, dtype=torch.int32)
        packed_indices = {
            "valid_token_indices_kv": valid_token_indices,
            "cu_seqlens_kv": cu_seqlens,
            "max_seqlen_in_batch_kv": total_len,
        }

        x = x.chunk(world_size, dim=1)[rank]

        # the rope is sliced along the head dimension
        cp_rank, cp_size = cp.get_cp_rank_size()
        local_heads = num_heads // cp_size
        rope_cos_local = rope_cos.narrow(1, cp_rank * local_heads, local_heads)
        rope_sin_local = rope_sin.narrow(1, cp_rank * local_heads, local_heads)
        out_forward = model._forward_original(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            packed_indices=packed_indices,
            rope_cos=rope_cos_local,
            rope_sin=rope_sin_local,
        )

        total_len = x.size(1) + valid_y_seq
        valid_token_indices = torch.arange(valid_y_seq, device=device)
        cu_seqlens = torch.tensor([0, total_len], device=device, dtype=torch.int32)
        packed_indices = {
            "valid_token_indices_kv": valid_token_indices, #? why its shape is not [total_len]
            "cu_seqlens_kv": cu_seqlens, # useless for xdit
            "max_seqlen_in_batch_kv": total_len, # useless for xdit
        }

        # NOTE() the input to rope is replicated
        rope_cos_local = rope_cos.chunk(world_size, dim=0)[rank]
        rope_sin_local = rope_sin.chunk(world_size, dim=0)[rank]
        out_xdit = model._forward_xdit(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            packed_indices=packed_indices,
            rope_cos=rope_cos_local,
            rope_sin=rope_sin_local,
        )

    # Compare outputs
    x_forward, y_forward = out_forward
    x_xdit, y_xdit = out_xdit

    # Check shapes match
    assert x_forward.shape == x_xdit.shape, f"X shape mismatch: {x_forward.shape} vs {x_xdit.shape}"
    assert y_forward.shape == y_xdit.shape, f"Y shape mismatch: {y_forward.shape} vs {y_xdit.shape}"

    # # Check values are close
    torch.testing.assert_close(x_forward, x_xdit, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(y_forward, y_xdit, rtol=1e-3, atol=1e-3) 

    print("Passed")

if __name__ == "__main__":
    test_forward_xdit_matches_forward()