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

import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from glob import glob
from numpy.lib.stride_tricks import as_strided as ast
import random, string, os
from joblib import Parallel, delayed

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))


ims_per_shard = 200

###==========================================================================

def writeout_tfrecords(storm):
    # for storm in ['matthew', 'michael', 'florence', 'harvey']:
    if storm=='matthew':
        n=842
    elif storm=='michael':
        n=1057
    elif storm=='florence':
        n=586
    elif storm=='harvey':
        n=1172

    print('Working on %s' % (storm))
    imdir = '/media/marda/TWOTB1/xBD/hurricanes/tiled_images/'+storm
    tfrecord_dir = '/media/marda/TWOTB1/xBD/hurricanes/tfrecords/'+storm+'/imrecog'

    nb_images=len(glob(imdir+os.sep+'destroyed/*.jpg'))+len(glob(imdir+os.sep+'no-damage/*.jpg'))+\
        len(glob(imdir+os.sep+'minor-damage/*.jpg'))+len(glob(imdir+os.sep+'major-damage/*.jpg'))+len(glob(imdir+os.sep+'un-classified/*.jpg'))
    print('Image tiles: %i' % (nb_images))

    SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)
    print('tfrecord shards: %i' % (SHARDS))

    shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

    all_images=glob(imdir+os.sep+'destroyed/*.jpg')+glob(imdir+os.sep+'no-damage/*.jpg')+\
        glob(imdir+os.sep+'minor-damage/*.jpg')+glob(imdir+os.sep+'major-damage/*.jpg')+glob(imdir+os.sep+'un-classified/*.jpg')

    for k in range(10):
        random.shuffle(all_images)
    Z,_ = sliding_window(np.array(all_images), (shared_size), (shared_size))

    for counter in range(n,len(Z)):
        try:
            print('%i out of %i' % (counter, len(Z)))
            dataset = tf.data.Dataset.list_files(Z[counter], shuffle=None) #imdir+os.sep+'destroyed/*.jpg',
            dataset = get_recog_dataset_for_tfrecords(dataset, shared_size)
            write_records(dataset, tfrecord_dir, types, counter)
        except:
            pass

#-----------------------------------
"""
These functions cast inputs into tf dataset 'feature' classes
There is one for bytestrings (images), one for floats (not used here) and one for ints (labels)
"""
def _bytestring_feature(list_of_bytestrings):
    """
    "_bytestring_feature(list_of_bytestrings)"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_bytestrings
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints):
    """
    "_int_feature(list_of_ints)"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_ints
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats):
    """
    "_float_feature(list_of_floats)"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_floats
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

#-----------------------------------
def to_tfrecord(img_bytes, label, types):
    """
    to_tfrecord(img_bytes, label, types)
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes: an image bytestring
        * label: label string of image
        * types: list of string classes in the entire dataset
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    class_num = np.argmax(np.array(types)==label)
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "class": _int_feature([class_num]),        # one class in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# =========================================================
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')

# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape

# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

    #l, cnt = md(cl.flatten())
    #l = np.squeeze(l)
    #if l==0:

    dist = np.bincount(cl.flatten(), minlength=len(labels))
    if np.all(dist[1:]==0)==True:
        l=0
        cnt = np.max(dist)
    else:
        l=np.argmax(dist[1:])+1
        cnt = np.max(dist[1:])

    if cnt/len(cl.flatten()) > thres:
        outfile = id_generator()+'.jpg'
        try:
            fp = outpath+os.sep+labels[l]+os.sep+outfile
            imwrite(fp, tmp)
        except:
            pass

#-----------------------------------
def to_tfrecord(img_bytes, label, CLASSES):
    """
    to_tfrecord(img_bytes, label, CLASSES)
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes: an image bytestring
        * label: label string of image
        * CLASSES: list of string classes in the entire dataset
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    class_num = np.argmax(np.array(CLASSES)==label)
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "class": _int_feature([class_num]),        # one class in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#-----------------------------------
def read_image_and_label(img_path):
    """
    read_image_and_label(img_path)
    This function reads a jpeg image from a provided filepath
    and extracts the label from the filename (assuming the class name is
    before "_IMG" in the filename)
    INPUTS:
        * img_path [string]: filepath to a jpeg image
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor int]
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    label = tf.strings.split(img_path, sep='/')
    #label = tf.strings.split(label[0], sep='_IMG')

    return image,label[-2]

#-----------------------------------
def resize_and_crop_image(image, label):
    """
    resize_and_crop_image(image, label)
    This function crops to square and resizes an image
    The label passes through unmodified
    INPUTS:
        * image [tensor array]
        * label [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [int]
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
    return image, label

#-----------------------------------
def recompress_image(image, label):
    """
    recompress_image(image, label)
    This function takes an image encoded as a byte string
    and recodes as an 8-bit jpeg
    Label passes through unmodified
    INPUTS:
        * image [tensor array]
        * label [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * label [int]
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label

#-----------------------------------
def get_recog_dataset_for_tfrecords(dataset, shared_size):
    """
    "get_recog_dataset_for_tfrecords"
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
    dataset = dataset.map(read_image_and_label)
    dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)
    dataset = dataset.map(recompress_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(shared_size)
    return dataset


#-----------------------------------
def write_records(tamucc_dataset, tfrecord_dir, types, counter):
    """
    write_records(tamucc_dataset, tfrecord_dir, types)
    This function writes a tf.data.Dataset object to TFRecord shards
    INPUTS:
        * tamucc_dataset [tf.data.Dataset]
        * tfrecord_dir [string] : path to directory where files will be written
        * CLASSES [list] of class string names
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (files written to disk)
    """
    for shard, (image, label) in enumerate(tamucc_dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+"xbDhurricanes-"+str(counter) + "-{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_tfrecord(image.numpy()[i],label.numpy()[i], types)
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))


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

types = [k for k in damage_dict]

BATCH_SIZE = 4

TARGET_SIZE = tile = 96
thres = 0.25

# for storm in ['matthew', 'michael', 'florence', 'harvey']:
#     print('Working on %s' % (storm))
#     imdir = '/media/marda/TWOTB1/xBD/hurricanes/images/'+storm
#     lab_path =  '/media/marda/TWOTB1/xBD/hurricanes/labels2D/'+storm
#     outpath = '/media/marda/TWOTB1/xBD/hurricanes/tiled_images/'+storm
#
#     images = sorted(glob(imdir+os.sep+'*.png'))
#     labels = sorted(glob(lab_path+os.sep+'*.png'))
#
#     for i,l in zip(images, labels):
#         Z,_ = sliding_window(imread(i), (tile,tile,3), (int(tile/2), int(tile/2),3))
#
#         C,_ = sliding_window(imread(l), (tile,tile), (int(tile/2), int(tile/2)))
#
#         Parallel(n_jobs=-1, verbose=0)(delayed(writeout)(Z[k], C[k], types, outpath, thres) for k in range(len(Z)))
#         del Z, C
#     del images, labels


Parallel(n_jobs=-1, verbose=0)(delayed(writeout_tfrecords)(storm) for storm in ['matthew', 'michael', 'florence', 'harvey'])

# storm='matthew'
#
# writeout_tfrecords(storm)






#==
