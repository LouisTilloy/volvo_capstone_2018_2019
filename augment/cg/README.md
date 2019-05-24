# CycleGAN (CG)
This image generation method utilizes CycleGAN for image-to-image tranformations.

## New training:
The dataset directory should be split into three folders: trainA, trainB and testA. 
* testA: day-time images for transformation from day to night
* trainA: day-time images
* trainB: night-time images

```
python train.py --dataroot ./datasets/DATASET_FOLDER --name day2night --model cycle_gan --display_id -1 --load_size 256
```

## Continue training:
```
python train.py --dataroot ./datasets/DATASET_FOLDER --name day2night --model cycle_gan --display_id -1 --load_size 256 --continue_train --epoch_count LATEST_EPOCH
```

## Genrate data:
```
python test.py --dataroot datasets/day2night/testA --name day2night --model test --no_dropout --num_test NO.IMAGES
```