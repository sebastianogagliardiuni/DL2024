# AIS DL 2024

Source code to generate the results of experiments for AIS DL 2024 exam.

## Dependencies

Install Torch with CUDA for GPU fast usage. 

```shell
pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

Addional mandatory dependencies

```shell
pip3 install tqdm==4.67.1
pip3 install scikit-learn==1.5.2
```

If you desire to log results on W&B install also wanb library.

```shell
pip3 install wandb
```
Remember to modify the "wandb-config.json" to link your W&B project.

## Datasets

Download the datasets.

```shell
mkdir ./datasets
mkdir ./datasets/ImageNet-A
mkdir ./datasets/ImageNet-V2
wget https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xf ./imagenet-a.tar
tar -xf ./imagenetv2-matched-frequency.tar.gz
mv imagenet-a ./datasets/ImageNet-A
mv imagenetv2-matched-frequency-format-val ./datasets/ImageNet-V2
```

## Usage

Here you can have a look at the arguments that each source file requires, remember that you can get more info with the tag "--help".

### Baseline

```shell
python3 ./src/test-baseline.py
    -m
    --adapt
    --prior_N
    --wandb
```

### MA

```shell
python3 ./src/test-multi-augmentation.py
    -m
    --adapt
    -a
    -N
    -ct
    --prior_N
    --wandb
```

### MEMO

```shell
python3 ./src/test-memo.py
    -m
    --adapt
    --use_batch_stats
    -a
    -N
    -ct
    -o
    --lr
    --prior_N
    --wandb
```

### RMEMO

#### KL

```shell
python3 ./src/test-kl-memo.py
    -m
    --adapt
    --use_batch_stats
    -a
    -N
    -K
    -bc
    -ct
    -o
    --lr
    --prior_N
    --wandb
```
#### SR

```shell
python3 ./src/test-tk-memo.py
    -m
    --adapt
    --use_batch_stats
    -a
    -N
    -K
    -bc
    -ct
    -o
    --lr
    --prior_N
    --wandb
```

#### SR

```shell
python3 ./src/test-score-memo.py
    -m
    --adapt
    --use_batch_stats
    -a
    -N
    -K
    -bc
    -ct
    -o
    --lr
    --prior_N
    --wandb
```

## Experiments

To reproduce experiments shown in the Notebook you can have a look at "Report.md" in "./experiments".





