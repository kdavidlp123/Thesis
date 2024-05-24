# Thesis
# Introduction
LENS-iAcu is an mobile application which can localize auricular acupoint in real-time scenario. This project is implemented based on OpenMMLab.
Here is the demo video of LENS-iAcu:<br>
https://youtu.be/qC5TXhiM7pM


# Environment Settings
* Windows 10
* Anaconda
* cuda version: 11.1
# Installation
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/kdavidlp123/Thesis.git
cd Thesis/mmdetection
pip install -v -e .
cd ../mmpose
pip install -r requirements.txt
pip install -v -e .
```
# Data Acquisition
Please contact us through the e-mail: P76091519@gs.ncku.edu.tw to acquire the data.








