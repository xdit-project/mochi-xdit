# mochi-xDiT
Optimized mochi video generation model

Install dependencies

```shell
sudo apt install ffmpeg
pip install .
```

Running mochi with a single GPU

```shell
CUDA_VISIBLE_DEVICES=0 python3 ./demos/cli.py --model_dir "<path_to_downloaded_directory>" --prompt "prompt"
```