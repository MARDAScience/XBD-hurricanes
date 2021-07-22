# Written by Dr Daniel Buscombe, Marda Science LLC
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


TARGET_SIZE = 1024
ims_per_shard = 20


#-----------------------------------
def get_seg_dataset_for_tfrecords(imdir, lab_path, shared_size):
    """
    "get_seg_dataset_for_tfrecords"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This is the version for data, which differs in use of both
    resize_and_crop_seg_image and resize_and_crop_seg_image
    for image pre-processing
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    dataset = tf.data.Dataset.list_files(imdir+os.sep+'*.png', seed=10000) # This also shuffles the images
    dataset = dataset.map(read_seg_image_and_label)
    dataset = dataset.map(resize_and_crop_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(shared_size)
    return dataset

#-----------------------------------
def read_seg_image_and_label(img_path):
    """
    "read_seg_image_and_label(img_path)"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This works by parsing out the label image filename from its image pair
    Thre are different rules for non-augmented versus augmented imagery
    INPUTS:
        * img_path [tensor string]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [bytestring]
        * label [bytestring]
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_png(bits)

    # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
    lab_path = tf.strings.regex_replace(img_path, "images", "labels2D")

    lab_path = tf.cast(lab_path, tf.string)
    #if not_contains(lab_path, "pre"): #tf.constant('post') in lab_name: # #tf.strings.regex_full_match(lab_path, tf.constant("post")): #
    lab_path = tf.strings.regex_replace(lab_path, "_post_disaster.png", "_post_disaster_post_labelimage.png")
    #else:
    lab_path = tf.strings.regex_replace(lab_path, "_pre_disaster.png", "_pre_disaster_pre_labelimage.png")

    lab_path = tf.cast(lab_path, tf.string)

    bits = tf.io.read_file(lab_path)
    label = tf.image.decode_png(bits)

    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*7)
    label = tf.where(cond,  tf.ones(tf.shape(label),dtype=tf.uint8)*6, label)

    return image, label

#-----------------------------------
def resize_and_crop_seg_image(image, label):
    """
    "resize_and_crop_seg_image"
    This function crops to square and resizes an image and label
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)

    label = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(label, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(label, [w*th/h, h*th/h])  # if false
                 )
    label = tf.image.crop_to_bounding_box(label, (nw - tw) // 2, (nh - th) // 2, tw, th)

    label = tf.cast(label, tf.uint8)
    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*7)
    label = tf.where(cond,  tf.ones(tf.shape(label),dtype=tf.uint8)*6, label)

    return image, label


#-----------------------------------
def recompress_seg_image(image, label):
    """
    "recompress_seg_image"
    This function takes an image and label encoded as a byte string
    and recodes as an 8-bit jpeg
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_png(image)#, optimize_size=True, chroma_downsampling=False)

    label = tf.cast(label, tf.uint8)

    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*7)
    label = tf.where(cond,  tf.ones(tf.shape(label),dtype=tf.uint8)*6, label)

    label = tf.image.encode_png(label)#, optimize_size=True, chroma_downsampling=False)

    return image, label

#-----------------------------------
def write_seg_records(dataset, tfrecord_dir, storm):
    """
    "write_seg_records(dataset, tfrecord_dir)"
    This function writes a tf.data.Dataset object to TFRecord shards
    The version for data preprends "" to the filenames, but otherwise is identical
    to write_seg_records
    INPUTS:
        * dataset [tf.data.Dataset]
        * tfrecord_dir [string] : path to directory where files will be written
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (files written to disk)
    """
    for shard, (image, label) in enumerate(dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+"hurricane-"+storm + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_seg_tfrecord(image.numpy()[i],label.numpy()[i])
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))

#-----------------------------------
def _bytestring_feature(list_of_bytestrings):
    """
    "_bytestring_feature"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_bytestrings
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

#-----------------------------------
def to_seg_tfrecord(img_bytes, label_bytes):
    """
    "to_seg_tfrecord"
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes
        * label_bytes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "label": _bytestring_feature([label_bytes]), # one label image in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#-----------------------------------
def seg_file2tensor(f):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """
    bits = tf.io.read_file(f)
    image = tf.image.decode_png(bits)

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    # image = tf.cast(image, tf.uint8) #/ 255.0

    return image



###############################################################
## VARIABLES
###############################################################
damage_dict = {
    "no-damage": (0, 255, 0),
    "minor-damage": (0, 0, 255),
    "major-damage": (255, 69, 0),
    "destroyed": (255, 0, 0),
    "un-classified": (255, 255, 255)
}
cols = [damage_dict[k] for k in damage_dict]

BATCH_SIZE = 4

for storm in ['matthew', 'michael', 'florence', 'harvey']:

    imdir = '/media/marda/TWOTB1/xBD/hurricanes/images/'+storm

    lab_path =  '/media/marda/TWOTB1/xBD/hurricanes/labels2D/'+storm

    tfrecord_dir = '/media/marda/TWOTB1/xBD/hurricanes/tfrecords/'+storm+'/imseg'


    ###############################################################
    ## EXECUTION
    ###############################################################

    nb_images=len(tf.io.gfile.glob(imdir+os.sep+'*.png'))

    SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)

    shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

    dataset = get_seg_dataset_for_tfrecords(imdir, lab_path, shared_size)

    ##view a batch
    for imgs,lbls in dataset.take(1):
      imgs = imgs[:BATCH_SIZE]
      lbls = lbls[:BATCH_SIZE]
      for count,(im,lab) in enumerate(zip(imgs,lbls)):
         lab = tf.image.decode_png(lab, channels=1)
         plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
         plt.imshow(tf.image.decode_png(im, channels=3))
         plt.imshow(lab, alpha=0.5, cmap='bwr')
         plt.axis('off')
         print(np.unique(lab.numpy().flatten()))
    plt.show()

    write_seg_records(dataset, tfrecord_dir, storm)
