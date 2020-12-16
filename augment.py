#Data Augmentation

#Import required libraries:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse

#Parsing the arguments:
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path of the input image')
ap.add_argument("-o", "--output", required=True,
help="path to output directory to store augmentation examples")
#ap.add_argument('-o', '--output' required=True, help="path of the output directory, where the augmented images will be saved")
ap.add_argument("-p", "--prefix", type=str, default="image",help="output filename prefix")
args = vars(ap.parse_args())

#Loading image:
print("[INFO] loading uploaded image....!")
# 1. Load the image from folder using arguments:
image = load_img(args['image'])
# 2. Convert loaded image to arrays:
image = img_to_array(image)
# 3. Expanding the dimensions using np:
image = np.expand_dims(image, axis=0)

# Constuct image data generator 

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                        shear_range = 0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
total=0


print("[INFO] generating images...")
#actual python code to generate the images:
imageGen = aug.flow(image, batch_size=2, save_to_dir=args["output"],
save_prefix=args["prefix"], save_format="jpg")

for image in imageGen:
    total += 1
#To generate 10 number of images:
    if total == 10:
        break


print("[INFO] Images generated! Check the output folder!!")


