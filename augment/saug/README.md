# SimpleAugment (SAUG)
This image transformation utilizes simple color manipulation to make day-time images look like night-time.

The script manipulates images using simple operations. The images are transformed in three steps. 
1. Increase the blue color value in each pixel and lower the RGB values in the image depending on the initial RGB value. Pixels with higher values get decreased exponentially more than pixels with initially lower values in order to create a darker version of the input image.
1. Further darkening of the pixels in the top half of the image to make the sky darker.
1. Attaching the traffic sign from the input image on the exact same location using the coordinates from the original bounding box but in a lighter tone than the rest of the image, in order to highlight the traffic sign.