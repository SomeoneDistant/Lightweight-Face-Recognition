# ResNet model with Arcface and Focal Loss

## Example of training
```
python main.py \
--train \
--model ResNet101 \
--batch_size 1024 \
--epoch_size 1000 \
--optim SGD \
--loss_function FocalLoss \
--data_path ./lite
```

## Example of testing
```
python main.py \
--inference \
--model ResNet101 \
--batch_size 512 \
--data_path ./lite
```

## Directory tree
```
.
├── main.py
├── model.py
├── dataset.py
├── lossfunction.py
├── _micro
|   ├── imglist_iccv.txt
|   └── _images
|       ├── _00000012
|       |   ├── 00000741.jpg
|       |   └── ...
|       └── _...
└── _lite
    ├── imglist_iccv.txt
    └── _images
        ├── _00000003
        |   ├── 00000065.jpg
        |   └── ...
        └── _...
```
