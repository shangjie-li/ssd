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
   ```
 - Prepare dataset
   ```
   # Assume that you have cloned this repository into SSD_ROOT.
   sh $SSD_ROOT/scripts/VOC2007.sh
   sh $SSD_ROOT/scripts/VOC2012.sh
   
   # By default, the dataset will be download to ~/data/VOCdevkit.
   # If you change the root directory of the dataset, please make corresponding adjustments in cfg/config.py.
   ```

## Training
 - Prepare pretrained model
   ```
   cd $SSD_ROOT
   mkdir weights && cd weights
   wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
   ```
 - Run the command below to train
   ```
   # Start training and specify batch_size and num_workers accordingly.
   python train.py --batch_size=16 --num_workers=4
   
   # Resume training and specify the model to start from.
   python train.py --resume=weights/ssd300_13_10000.pth
   ```
 
## Evaluation
 - A trained model is provided [here](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth).
 - Run the command below to evaluate
   ```
   # Evaluate on PASCAL VOC dataset and specify the model.
   python eval.py --trained_model=weights/ssd300_mAP_77.43_v2.pth
   
   # Display the prediction results and filter the results by setting conf_thresh and top_k.
   python eval.py --display=True --conf_thresh=0.5 --top_k=20
   ```
