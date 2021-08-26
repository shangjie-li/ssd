# ssd

Implementation of SSD in PyTorch for PASCAL VOC detection

## Acknowledgement
 - This repository references [amdegroot](https://github.com/amdegroot/ssd.pytorch)'s work.

## Installation
 - Install PyTorch environment with Anaconda (Test with CUDA 9.0)
   ```
   conda create -n ssd python=3.6
   conda activate ssd
   conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch
   pip install opencv-python matplotlib
   ```
 - Clone this repository
   ```
   git clone https://github.com/shangjie-li/ssd.git
   
   # Assume that you clone this repository into SSD_ROOT
   ```
 - Prepare dataset
   ```
   # By default, the dataset will be download to ~/data/VOCdevkit
   
   sh $SSD_ROOT/data/scripts/VOC2007.sh
   sh $SSD_ROOT/data/scripts/VOC2012.sh
   
   # If you change root directory of the dataset, please adjust VOC_ROOT in cfg/config.py
   ```

## Training
 - Prepare pretrained weight
   ```
   cd $SSD_ROOT
   mkdir weights && cd weights
   wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
   ```
 - Run the command below
   ```
   python train.py
   python train.py --batch_size=16 --num_workers=4
   python train.py --batch_size=16 --num_workers=4 --resume=ssd300_mAP_77.43_v2.pth
   ```
 
## Evaluation
 - Run the command below
   ```
   python eval.py
   python eval.py --trained_model=ssd300_mAP_77.43_v2.pth --conf_thresh=0.6 --top_k=20 --nms_thresh=0.5
   python eval.py --trained_model=ssd300_mAP_77.43_v2.pth --conf_thresh=0.6 --top_k=20 --nms_thresh=0.5 --display=True
   ```

## Application
 - Run the command below
   ```
   python ssd_detector.py
   ```