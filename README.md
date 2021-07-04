# Versatile-Weight-Attack
The implementatin of *Versatile Weight Attack via Flipping Limited Bits*, including *Singel Sample Attack* and *Triggered Samples Attack*.


## Install 
1. Install PyTorch >= 1.5
2. Clone this repo:
```shell
git clone https://github.com/jiawangbai/Versatile-Weight-Attack.git
```

## Run Our Code

### Single Sample Attack
Set the "cifar_root" in the "config.py" firstly.

Running the below command will attack a sample (3676-th sample in the CIFAR-10 validation set) into class 0.
```shell
python attack_one.py --target-class 0 --attack-idx 3676 --lam 100 --k 5
```
You can set "target-class" and "attack-idx" to perform TA-LBF on a specific sample.

## Reproduce Our Results
Set the "cifar_root" in the "config.py" firstly.

Running the below command can reproduce our results in attacking the 8-bit quantized ResNet on CIFAR-10 with the parameter searching strategy introduced in the paper.
```shell
python attack_reproduce.py 
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


### Triggered Samples Attack

## Others
We provide the pretrained 8-bit quantized ResNet on CIFAR-10. -> "cifar_resnet_quan_8/model.th"

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
