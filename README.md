# mochi-xdit: Parallel Inference for Mochi-preview Video Generation Model with xDiT

This repository provides an accelerated inference version of [Mochi 1](https://github.com/genmoai/models) using Unified Sequence Parallelism provided by [xDiT](https://github.com/xdit-project/xDiT).

Mochi-1 originally ran it on 4xH100 (100GB VRAM) GPUs, however, we made it run on a single 48GB L40 GPU with no accuracy loss!

## HightLights

1. Memory Optimization makes mochi is able to generate video on a single 48GB L40 GPU without no accuracy loss.
2. Tiled VAE decoder enables the correct generation of video with any resolution.
2. Unified Sequence Parallelism for AsymmetricAttention using xDiT: hybrid 2D sequence parallelism with Ring-Attention and DeepSpeed-Ulysses.

## Usage

This repository provides an accelerated inference version of [Mochi 1](https://github.com/genmoai/models) using Unified Sequence Parallelism provided by [xDiT](https://github.com/xdit-project/xDiT).

<div align="center">

| Feature | xdit version | original version|
|:---:|:---:|:---:|
| attention parallel | USP(Ulysses+Ring) | Ulysses |
| VAE | input tiling | X |
| model loading | Replicated | FSDP |

</div>

| Feature | xdit | original |
|:---:|:---:|:---:|
| attention parallel | USP(Ulysses+Ring) | Ulysses |
| VAE | input titling | X |
| model loading | Replicated | FSDP |


## Usage

### 1. Install from source

```shell
pip install xfuser
sudo apt install ffmpeg
pip install .
```

### 2. Install from docker

```shell
docker pull thufeifeibear/mochi-dev:0.1
```

### 3. Run

Running mochi with a single GPU

```shell
CUDA_VISIBLE_DEVICES=0 python3 ./demos/cli.py --model_dir "<path_to_downloaded_directory>" --prompt "prompt"
```

Running mochi with multiple GPUs using Unified Sequence Parallelism provided by [xDiT](https://github.com/xdit-project/xDiT).

Use the number of GPUs in CUDA_VISIBLE_DEVICES to control world_size.

Adjust the configuration of ulysses_degree and ring_degree to achieve optimal performance. ulysses_degree x ring_degree = world_size.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 ./demos/cli.py --model_dir "<path_to_downloaded_directory>" --prompt "prompt" \
 --use_xdit --ulysses_degree 2 --ring_degree 2
```

### 4. Performance

<div align="center">

| Configuration | Metric | 1x L40 (baseline) | 2x L40 ||| 6x L40 |||
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | | U2 | R2 | baseline2 | u2r3 | u6 | baseline6 |
| **With FSDP** | Sampling (s) | 388.61 | 257.71 | 258.19 | 257.7 | 470.99 | 474.66 | 471.09 |
| | Conditioning (s) | 0.5 | 0.92 | 1.0 | 0.88 | 1.22 | 1.42 | 1.18 |
| | VAE Decoding (s) | 7.84 | 11.59 | 10.47 | 10.47 | 9.22 | 9.19 | 9.15 |
| | Memory (GB) | 35.43 | 38.84 | 38.84 | 37.71 | 19.54 | 16.33 | 19.7 |
| **Without FSDP** | Sampling (s) | 396.04 | 216.45 | 216.7 | 204.5 | 242.51 | 246.66 | 242.34 |
| | Conditioning (s) | 0.37 | 0.94 | 0.96 | 1.47 | 1.25 | 1.2 | 1.21 |
| | VAE Decoding (s) | 7.81 | 10.3 | 10.59 | 10.18 | 9.81 | 10.04 | 9.2 |
| | Memory (GB) | 35.43 | 37.97 | 38.42 | 33.61 | 30 | 29.07 | 29.99 |

</div>

### References

[xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism](https://arxiv.org/abs/2411.01738)

```
@misc{fang2024xditinferenceenginediffusion,
      title={xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism}, 
      author={Jiarui Fang and Jinzhe Pan and Xibo Sun and Aoyu Li and Jiannan Wang},
      year={2024},
      eprint={2411.01738},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2411.01738}, 
}
```

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)

```
@misc{fang2024uspunifiedsequenceparallelism,
      title={USP: A Unified Sequence Parallelism Approach for Long Context Generative AI}, 
      author={Jiarui Fang and Shangchun Zhao},
      year={2024},
      eprint={2405.07719},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.07719}, 
}
```

