# Versatile-Weight-Attack
The implementatin of *Versatile Weight Attack via Flipping Limited Bits*, including *Singel Sample Attack* and *Triggered Samples Attack*.

## Updates
[2023/05/01] The attacks against full-precision netoworks (their weights are
in 32-bit floating point format) have been added to the folders "single_sample_attack_float" and "triggered_samples_attack_float".

## Install 
1. Install PyTorch >= 1.5
2. Clone this repo:
```shell
git clone https://github.com/jiawangbai/Versatile-Weight-Attack.git
```

## Run Our Code

Set the "cifar_root" in the "config.py" firstly.

### Single Sample Attack (SSA)

Running the below command will attack a sample (3676-th sample in the CIFAR-10 validation set) into class 0.
```shell
cd single_sample_attack/
python SSA.py --target-class 0 --attack-idx 3676 --lam 100 --k 5
```
You can set "target-class" and "attack-idx" to perform SSA on a specific sample.

Running the below command can reproduce our results in attacking the 8-bit quantized ResNet on CIFAR-10 with the parameter searching strategy introduced in the paper.
```shell
cd single_sample_attack/
python SSA_reproduce.py 
```
"cifar_attack_info.txt" includes the 1,000 attacked samples and their target classes used in our experiments.
<br/>
Format:
<br/>
&emsp; [[target-class, sample-index],
<br/>
&emsp; [target-class, sample-index],
<br/>
&emsp; ...
<br/>
&emsp; [target-class, sample-index] ]
<br/>
where "sample-index" is the index of this attacked sample in CIFAR-10 validation set.

Running the below command will attack the full-precision network.
```shell
cd single_sample_attack_float/
python SSA.py --target-class 0 --attack-idx 3676 --lam 50 --k 5
```

### Triggered Samples Attack (TSA)

Running the below command will attack all samples with a trigger (CIFAR-10 validation set) into class 0.
```shell
cd triggered_samples_attack/
python TSA.py --target 0
```
You can set "target" to perform TSA with different target class.


Running the below command will reproduce our results in attacking the 8-bit quantized ResNet on CIFAR-10 with the parameter searching strategy introduced in the paper.
```shell
cd triggered_samples_attack/
sh run.sh [gpu-id]
```

Running the below command will attack the full-precision network.
```shell
cd triggered_samples_attack_float/
python TSA.py --target 0
```

## Others
We provide the pretrained 8-bit quantized ResNet on CIFAR-10. -> "cifar_resnet_quan_8/model.th" and the pretrained full-precision ResNet on CIFAR-10. -> "cifar_resnet_float/model.th"

Python version is 3.6.10 and the main requirments are below:
<br/>
&emsp; torch==1.5.0
<br/>
&emsp; bitstring==3.1.7
<br/>
&emsp; torchvision==0.6.0a0+82fd1c8
<br/>
&emsp; numpy==1.18.1

We also provide the following command to install dependencies before running the code:
```shell
pip install -r requirements.txt
```
