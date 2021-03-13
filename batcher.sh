#!/bin/sh
set -e

SEED=$1

echo "Training CIFAR100"
python main.py --model-name="VGG16" --dataset-name="cifar100" --seed=655321
python main.py --model-name="Conv6" --dataset-name="cifar100" --seed=655321
python main.py --model-name="Conv2" --dataset-name="cifar100" --seed=655321
python main.py --model-name="FC"    --dataset-name="cifar100" --seed=655321

echo "Training CIFAR10"
python main.py --model-name="VGG16" --dataset-name="cifar10" --seed=655321
python main.py --model-name="Conv6" --dataset-name="cifar10" --seed=655321
python main.py --model-name="Conv2" --dataset-name="cifar10" --seed=655321
python main.py --model-name="FC"    --dataset-name="cifar10" --seed=655321

echo "Training MNIST"
python main.py --model-name="VGG16" --dataset-name="mnist" --seed=655321
python main.py --model-name="Conv6" --dataset-name="mnist" --seed=655321
python main.py --model-name="Conv2" --dataset-name="mnist" --seed=655321
python main.py --model-name="FC"    --dataset-name="mnist" --seed=655321

tar -zcf $SEED.tar.gz models/*$SEED