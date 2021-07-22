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
BATCH_SIZE = 4

@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_tfrecord_multiclass(example):
    """
    "read_seg_tfrecord_multiclass(example)"
    This function reads an example from a TFrecord file into a single image and label
    This is the "multiclass" version for imagery, where the classes are mapped as follows:
    INPUTS:
        * TFRecord example object
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_png(example['image'], channels=3)
    image = tf.cast(image, tf.float32)/ 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
    #image = tf.reshape(tf.image.rgb_to_grayscale(image), [TARGET_SIZE,TARGET_SIZE, 1])

    label = tf.image.decode_png(example['label'], channels=1)
    label = tf.cast(label, tf.uint8)#/ 255.0
    label = tf.reshape(label, [TARGET_SIZE,TARGET_SIZE, 1])

    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*7)
    label = tf.where(cond,  tf.ones(tf.shape(label),dtype=tf.uint8)*6, label)

    label = tf.one_hot(tf.cast(label, tf.uint8), 6) #6 = 5 classes (undamaged, minor, major, destroyed, unclass) + null (0)

    label = tf.squeeze(label)

    image = tf.reshape(image, (image.shape[0], image.shape[1], image.shape[2]))

    #image = tf.image.per_image_standardization(image)
    return image, label

#-----------------------------------
def get_batched_dataset(filenames):
    """
    "get_batched_dataset(filenames)"
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    INPUTS:
        * filenames [list]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: BATCH_SIZE, AUTO
    OUTPUTS: tf.data.Dataset object
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_seg_tfrecord_multiclass, num_parallel_calls=AUTO)
    #dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    #dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

# from tensorflow.python.client import device_lib
#
# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']


#==================================================================
for storm in ['matthew', 'michael', 'florence', 'harvey']:

    imdir = '/media/marda/TWOTB1/xBD/hurricanes/images/'+storm

    lab_path =  '/media/marda/TWOTB1/xBD/hurricanes/labels2D/'+storm

    tfrecord_dir = '/media/marda/TWOTB1/xBD/hurricanes/tfrecords/'+storm+'/imseg'

    # # Run inference on CPU
    # with tf.device('/cpu:0'):

    ##test
    filenames = sorted(tf.io.gfile.glob(tfrecord_dir+'/*.jpg'))
    dataset = get_batched_dataset(filenames)

    B = []
    for imgs,lbls in dataset.take(1):
      for count,(im,lab) in enumerate(zip(imgs,lbls)):
         print(np.shape(lab))
         lab= np.argmax(lab,axis=-1)
         B.append(np.bincount(lab.flatten(),minlength=6))
         plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
         plt.imshow(im)
         del im
         plt.imshow(lab, alpha=0.5, cmap='bwr')
         plt.axis('off')
         del lab
    plt.show()
    np.sum(np.vstack(B),axis=0)
