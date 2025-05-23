# MAIN

## Baseline

```shell
python ./src/test-baseline.py -m="ResNet-50" --wandb
python ./src/test-baseline.py -m="ResNet-50" --adapt --wandb
```

## MA

```shell
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 --adapt --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 --adapt --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="Mixture" -N=64 --adapt --wandb
```

## MA - CT

```shell
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5  --adapt --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 --adapt --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 --adapt --wandb
```

## MEMO

```shell
python ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -o="SGD" --lr="0.00025" --wandb
python ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -o="SGD" --lr="0.00025"  --adapt --wandb
python ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -o="SGD" --lr="0.00025" --wandb
python ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-memo.py -m="ResNet-50" -a="Mixture" -N=64 -o="SGD" --lr="0.00025" --adapt --wandb
```

## MEMO - CT

```shell
python ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --wandb
python ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025"  --adapt --wandb
python ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --wandb
python ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
```

## RMEMO - CT

### KL

```shell
python ./src/test-kl-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-kl-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-kl-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

### TK

```shell
python ./src/test-topk-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-topk-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-topk-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

### SR

```shell
python ./src/test-score-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-score-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-score-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

## Biased RMEMO - CT

```shell
python ./src/test-kl-memo.py -m="ResNet-50" -a="Mixture" -N=64 -bc=0.76 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-topk-memo.py -m="ResNet-50" -a="Mixture" -N=64 -bc=0.76 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-score-memo.py -m="ResNet-50" -a="Mixture" -N=64 -bc=0.76 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

# Ablations

## MA + CT = RMEMO (TK) + CT

```shell
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.125  --adapt --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.125 --adapt --wandb
python ./src/test-multi-augmentation.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.125 --adapt --wandb
```

# Additional results

## RMEMO Single image (K=1)

```shell
python ./src/test-kl-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-kl-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb 

python ./src/test-topk-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb 
python ./src/test-topk-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb

python ./src/test-score-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
python ./src/test-score-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb 
```

### Batch NA

```shell
python ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -o="SGD" --lr="0.00025" --adapt --use_batch_stats --wandb
python ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -o="SGD" --lr="0.00025" --adapt --use_batch_stats --wandb
python ./src/test-memo.py -m="ResNet-50" -a="Mixture" -N=64 -o="SGD" --lr="0.00025" --adapt --use_batch_stats --wandb
```