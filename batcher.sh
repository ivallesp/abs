#!/bin/sh
set -e

SEED=$1

echo "Training CIFAR100"
python main.py --model-name="VGG16" --dataset-name="cifar100" --seed=$SEED
python main.py --model-name="Conv6" --dataset-name="cifar100" --seed=$SEED
python main.py --model-name="Conv2" --dataset-name="cifar100" --seed=$SEED
python main.py --model-name="FC"    --dataset-name="cifar100" --seed=$SEED

echo "Training CIFAR10"
python main.py --model-name="VGG16" --dataset-name="cifar10" --seed=$SEED
python main.py --model-name="Conv6" --dataset-name="cifar10" --seed=$SEED
python main.py --model-name="Conv2" --dataset-name="cifar10" --seed=$SEED
python main.py --model-name="FC"    --dataset-name="cifar10" --seed=$SEED

echo "Training MNIST"
python main.py --model-name="VGG16" --dataset-name="mnist" --seed=$SEED
python main.py --model-name="Conv6" --dataset-name="mnist" --seed=$SEED
python main.py --model-name="Conv2" --dataset-name="mnist" --seed=$SEED
python main.py --model-name="FC"    --dataset-name="mnist" --seed=$SEED

tar -zcf $SEED.tar.gz models/*$SEED