# BoundingBoxGAN (BBGAN)
BBG is a GAN for image-to-image transition using a bounding box specific loss function. It can input images with annotated bounding boxes and transform these images in such way that the contents of the bounding box in the image is preserved.

## ```create.py```
This script is used to create a new GAN model. The model consists of a generator model and a discriminator model. 

## ```train.py```
This script is used to train the GAN model. To train the model it requires an input dataset of images to be tranformed and a real dataset with images of the desired style to transform to.

## ```infer.py```
This script is used on a trained GAN model to transform images.