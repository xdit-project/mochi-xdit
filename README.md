# mochi-xdit: Parallel Inference for Mochi-preview Video Generation Model with xDiT

This repository provides an accelerated inference version of [Mochi 1](https://github.com/genmoai/models) using Unified Sequence Parallelism provided by [xDiT](https://github.com/xdit-project/xDiT).

## HightLights

1. Memory Optimization makes mochi is  able to generate video on a single 48-GB GPU.
2. Tiling VAE optimization: makes mochi generate video on 48 GB GPU.
3. Unified Sequence Parallelism for AsymmetricAttention using xDiT: hybrid 2D sequence parallelism with Ring-Attention and DeepSpeed-Ulysses.


## Usage

This repository provides an accelerated inference version of [Mochi 1](https://github.com/genmoai/models) using Unified Sequence Parallelism provided by [xDiT](https://github.com/xdit-project/xDiT).

## HightLights

1. Memory Optimization makes mochi is  able to generate video on a single 48-GB GPU.
2. Tiling VAE optimization: makes mochi generate video on 48 GB GPU.
3. Unified Sequence Parallelism for AsymmetricAttention using xDiT: hybrid 2D sequence parallelism with Ring-Attention and DeepSpeed-Ulysses.


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

