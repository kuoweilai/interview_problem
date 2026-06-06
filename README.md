
## Environment setup
I run the code in Ubuntu 22.04.

virtual environment:
```
conda env create --name [MY_ENV] -f environment.yml
```

## Dataset

https://universe.roboflow.com/yaid-pzikt/firefighting-device-detection

YOLO26

## Code summary

1. preprocess object detection dataset into self-supervised representation learning dataset. Crop every object using yolo labels

2. create custon dataloader with random data augumentation.

3. implement contrastive loss and train a simple simclr model

4. plot train and validation loss.
