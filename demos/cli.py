#! /usr/bin/env python
import json
import os
import time

import click
import numpy as np
import torch

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)
from genmo.mochi_preview.dit.joint_model.globals import set_t5_model, set_max_t5_token_length, set_use_fsdp, set_use_xdit, set_usp_config

pipeline = None
model_dir_path = None
num_gpus = torch.cuda.device_count()
cpu_offload = False
dtype = None

def configure_model(model_dir_path_, cpu_offload_, 
        dtype_, use_xdit_, ulysses_degree_, ring_degree_, use_fsdp_, t5_model_path_, max_t5_token_length_):
    global model_dir_path, cpu_offload, dtype
    model_dir_path = model_dir_path_
    cpu_offload = cpu_offload_
    dtype = dtype_
    use_xdit = use_xdit_
    ulysses_degree = ulysses_degree_
    ring_degree = ring_degree_
    use_fsdp = use_fsdp_
    t5_model_path = t5_model_path_
    max_t5_token_length = max_t5_token_length_
    
    set_use_xdit(use_xdit)
    set_usp_config(ulysses_degree, ring_degree)
    set_use_fsdp(use_fsdp)
    set_t5_model(t5_model_path)
    set_max_t5_token_length(max_t5_token_length)

def load_model():
    global num_gpus, pipeline, model_dir_path
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(
                dtype=dtype,
            ),
            dit_factory=DitModelFactory(
                model_path=f"{MOCHI_DIR}/dit.safetensors", 
                model_dtype="bf16",
                dtype=dtype,
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/decoder.safetensors",
                dtype=dtype,
            ),
        )
        if num_gpus > 1:
            assert not cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = num_gpus
        else:
            kwargs["cpu_offload"] = cpu_offload
        kwargs["decode_type"] = "tiled_full"
        pipeline = klass(**kwargs)


def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
):
    load_model()

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_inference_steps,
        # We *need* flash attention to batch cfg
        # and it's only worth doing in a high-memory regime (assume multiple GPUs)
        "batch_cfg": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }

    with progress_bar(type="tqdm"):
        final_frames = pipeline(**args)

        final_frames = final_frames[0]

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", f"output_{int(time.time())}.mp4")


        save_video(final_frames, output_path)
        json_path = os.path.splitext(output_path)[0] + ".json"
        json.dump(args, open(json_path, "w"), indent=4)

        return output_path

from textwrap import dedent

DEFAULT_PROMPT = dedent("""
A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl 
filled with lemons and sprigs of mint against a peach-colored background. 
The hand gently tosses the lemon up and catches it, showcasing its smooth texture. 
A beige string bag sits beside the bowl, adding a rustic touch to the scene. 
Additional lemons, one halved, are scattered around the base of the bowl. 
The even lighting enhances the vibrant colors and creates a fresh, 
inviting atmosphere.
""")

@click.command()
@click.option("--prompt", default=DEFAULT_PROMPT, help="Prompt for video generation.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=49, type=int, help="Number of frames.")
@click.option("--seed", default=1710977262, type=int, help="Random seed.")
@click.option("--cfg_scale", default=4.5, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
@click.option("--use_xdit", is_flag=True, help="Whether to use xDiT")
@click.option("--ulysses_degree", default=None, type=int, help="Ulysses degree")
@click.option("--ring_degree", default=None, type=int, help="Ring degree")
@click.option("--use_fsdp", is_flag=True, help="Whether to use FSDP")
@click.option("--t5_model_path", default="/cfs/dit/t5-v1_1-xxl", type=str, help="the path of t5 model")
@click.option("--max_t5_token_length", default=256, type=int, help="the max token length of t5")
def generate_cli(
    prompt, negative_prompt, width, height, num_frames, seed, 
    cfg_scale, num_steps, model_dir, cpu_offload, use_xdit, ulysses_degree, ring_degree, use_fsdp, t5_model_path, max_t5_token_length   
):
    configure_model(model_dir, cpu_offload, torch.bfloat16, use_xdit, ulysses_degree, 
                   ring_degree, use_fsdp, t5_model_path, max_t5_token_length)
    output = generate_video(
        prompt,
        negative_prompt,
        width,
        height,
        num_frames,
        seed,
        cfg_scale,
        num_steps,
    )
    click.echo(f"Video generated at: {output}")


if __name__ == "__main__":
    generate_cli()
