#!/usr/bin/env bash
echo "Attack 8-bit quantized ResNet on CIFAR-10"

for target in 0 1 2 3 4 5 6 7 8 9
do
    python TSA.py --target $target --gpu-id $1
    wait
done
