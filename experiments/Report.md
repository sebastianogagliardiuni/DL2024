# MAIN

## Baseline

```shell
python3 ./src/test-baseline.py -m="ResNet-50" --wandb
python3 ./src/test-baseline.py -m="ResNet-50" --adapt --wandb
```

## MA

```shell
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 --adapt --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 --adapt --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="Mixture" -N=64 --adapt --wandb
```

## MA - CT

```shell
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5  --adapt --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 --adapt --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 --adapt --wandb
```

## MEMO

```shell
python3 ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -o="SGD" --lr="0.00025" --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -o="SGD" --lr="0.00025"  --adapt --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -o="SGD" --lr="0.00025" --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="Mixture" -N=64 -o="SGD" --lr="0.00025" --adapt --wandb
```

## MEMO - CT

```shell
python3 ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025"  --adapt --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
```

## RMEMO - CT

### KL

```shell
python3 ./src/test-kl-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-kl-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-kl-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

### TK

```shell
python3 ./src/test-topk-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-topk-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-topk-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

### SR

```shell
python3 ./src/test-score-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-score-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-score-memo.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

## Biased RMEMO - CT

```shell
python3 ./src/test-kl-memo.py -m="ResNet-50" -a="Mixture" -N=64 -bc=0.76 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-topk-memo.py -m="ResNet-50" -a="Mixture" -N=64 -bc=0.76 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-score-memo.py -m="ResNet-50" -a="Mixture" -N=64 -bc=0.76 -ct=0.5 -K=8 -o="SGD" --lr="0.00025" --adapt --wandb
```

# Ablations

## MA + CT = RMEMO (TK) + CT

```shell
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.125  --adapt --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.125 --adapt --wandb
python3 ./src/test-multi-augmentation.py -m="ResNet-50" -a="Mixture" -N=64 -ct=0.125 --adapt --wandb
```

# Additional results

## RMEMO Single image (K=1)

```shell
python3 ./src/test-kl-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-kl-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb 

python3 ./src/test-topk-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb 
python3 ./src/test-topk-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb

python3 ./src/test-score-memo.py -m="ResNet-50" -a="AugMix" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb
python3 ./src/test-score-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -ct=0.5 -o="SGD" --lr="0.00025" --adapt --wandb 
```

### Batch NA

```shell
python3 ./src/test-memo.py -m="ResNet-50" -a="AugMix" -N=64 -o="SGD" --lr="0.00025" --adapt --use_batch_stats --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="RandomResizedCrop" -N=64 -o="SGD" --lr="0.00025" --adapt --use_batch_stats --wandb
python3 ./src/test-memo.py -m="ResNet-50" -a="Mixture" -N=64 -o="SGD" --lr="0.00025" --adapt --use_batch_stats --wandb
```