# ResNet model with Arcface and Focal Loss

## Update log
- [x] Data augmentation
- [x] Pseudo fully connection
- [ ] Intermediate loss
- [ ] Octave conlolution
- [ ] Attention module
- [ ] Distillation
- [ ] Intrinsic part-based model
- [ ] Kernel orthogonalization

## Directory tree
```
.
├── main.py
├── model.py
├── dataset.py
├── lossfunction.py
├── augmentation.py
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
--pseudo
```

## Example of testing
```
python main.py \
--inference \
--model ResNet101 \
--batch_size 512 \
--data_path ./lite
--pseudo
```
