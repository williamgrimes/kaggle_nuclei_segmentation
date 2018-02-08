import os
import skimage

from skimage.color import rgba2rgb
from skimage.io import imread, imsave
from skimage.segmentation import mark_boundaries


file_path = './input/stage1_train/'
images = sorted(os.listdir(file_path))

for idx, image in enumerate(images):
    image_id = image
    image_dir = file_path + image_id + "/images/" + \
                image_id + ".png"
    image = skimage.io.imread(image_dir)
    image = rgba2rgb(image)

    masks = os.listdir(file_path + image_id + "/masks/")

    image_overlay = image
    for mask in masks:
        mask_name = file_path + image_id + "/masks/" + mask
        mask = skimage.io.imread(mask_name)
        image_overlay = mark_boundaries(image_overlay,
                                        mask, color=(0.6,0,0),
                                        outline_color=None,
                                        mode='outer')

    imsave('ground_truth_annotation/' + image_id + '.png',
           image_overlay)

    print('saved image %d of %d, image: %s \n' % \
          (idx + 1, len(images), image_id))
