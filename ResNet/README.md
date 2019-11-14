# ResNet model with Arcface and Focal Loss

## To-do list
- [x] Warm-up
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
├── README.md
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
--batch_size 384 \
--epoch_size 100 \
--optim SGD \
--loss_function FocalLoss \
--data_path ./lite \
--model ResNet50 \
--pseudo
```

## Example of testing
```
python main.py \
--inference \
--model ResNet50 \
--batch_size 384 \
--data_path ./lite \
--pseudo
```
