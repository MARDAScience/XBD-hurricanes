# Written by Dr Daniel Buscombe, Marda Science LLC
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

###############################################################
## VARIABLES
###############################################################

import tensorflow as tf
import os, io
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import pandas as pd
from collections import namedtuple

## conda install scikit-image
from skimage.measure import regionprops, label
from skimage.util import img_as_ubyte
from skimage.morphology import binary_dilation, disk

SEED = 2020
tf.random.set_seed(SEED)
np.random.seed(SEED)
AUTO = tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##========
def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, features)

##========
def write_records(csv_file,ims_per_shard,output_path, flag=99):
    examples = pd.read_csv(csv_file)
    print('Number of labels: %i' % len(examples))
    grouped = split(examples, 'filename')

    nb_images=len(grouped)
    print('Number of images: %i' % nb_images)

    ##758 images
    ##34620 individual buildings

    SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)
    print(SHARDS)

    shared_size = int(np.ceil(1.0 * nb_images / SHARDS))
    print(shared_size)

    # create indices into grouped that will enable writing SHARDS files, each containing shared_size examples
    grouped_forshards = np.lib.stride_tricks.as_strided(np.arange(len(grouped)), (SHARDS, shared_size))

    counter= 0
    for indices in grouped_forshards[:-1]:

        tmp = []
        for i in indices:
            tmp.append(grouped[i])

        if flag==1:
            op = output_path.replace('.tfrecord','')+ "buildings-{:02d}-{}.tfrec".format(counter, shared_size)
        elif flag==2:
            op = output_path.replace('.tfrecord','')+ "nodamage-damage-{:02d}-{}.tfrec".format(counter, shared_size)
        else:
            op = output_path.replace('.tfrecord','')+ "damage-{:02d}-{}.tfrec".format(counter, shared_size)
        writer = tf.io.TFRecordWriter(op)

        for group in tmp:
            tf_example = create_tf_example(group)
            writer.write(tf_example.SerializeToString())

        writer.close()
        print('Successfully created the TFRecords: {}'.format(op))

        counter += 1

##========
def split(df, group):
    """
    split(df, group)
    ""
    This function splits a pandas dataframe by a pandas group object
    to extract the label sets from each image
    for writing to tfrecords
    INPUTS:
        * df [pandas dataframe]
        * group [pandas dataframe group object]
    OPTIONAL INPUTS: None
    OUTPUTS:
        * tuple of bboxes and classes per image
    GLOBAL INPUTS: BATCH_SIZE
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

#===================================
def create_tf_example(group):
    """
    create_tf_example(group)
    ""
    This function creates an example tfrecord consisting of an image and label encoded as bytestrings
    The jpeg image is read into a bytestring, and the bbox coordinates and classes are collated and
    converted also
    INPUTS:
        * group [pandas dataframe group object]
        * path [tensorflow dataset]: training dataset
    OPTIONAL INPUTS: None
    OUTPUTS:
        * tf_example [tf.train.Example object]
    GLOBAL INPUTS: BATCH_SIZE
    """
    with tf.io.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    #encoded_jpg_io = io.BytesIO(encoded_jpg)

    filename = group.filename.encode('utf8')

    ids = []; areas = []; labels = []
    xmins = [] ; xmaxs = []; ymins = []; ymaxs = []

    #for converting back to integer
    class_dict = {'class1':0}

    for index, row in group.object.iterrows():
        #labels.append(class_dict[row['class']])
        labels.append(row['class'])
        ids.append(index)
        xmins.append(row['xmin'])
        ymins.append(row['ymin'])
        xmaxs.append(row['xmax'])
        ymaxs.append(row['ymax'])
        areas.append(row['area'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': bytes_feature(filename),
        'image/id': int64_list_feature(ids),
        'image': bytes_feature(encoded_jpg),
        'objects/xmin': float_list_feature(xmins), #xs
        'objects/xmax': float_list_feature(xmaxs), #xs
        'objects/ymin': float_list_feature(ymins), #xs
        'objects/ymax': float_list_feature(ymaxs), #xs
        'objects/area': float_list_feature(areas), #ys
        'objects/id': int64_list_feature(ids), #ys
        'objects/label': int64_list_feature(labels),
    }))

    return tf_example

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

##========
def get_bboxes(lab_path, cols, types, prc_increase_aoi):
    images = tf.io.gfile.glob(lab_path+os.sep+'*.png')

    F1 = []; B1 = []; M1 = []; m1 = []; A1 = []; N1 = []; L1 = []

    for i in images:
        img = imread(i)
        img = img_as_ubyte(img)

        for c,t in zip(cols,types):
            mask = np.all(img==c,axis=2)
            #mask = binary_dilation(mask, disk(5))
            label_img = label(mask, connectivity=mask.ndim)
            del mask
            props = regionprops(label_img)
            N1.append(len(props))
            for p in props:
                F1.append(i)
                bbox = p.bbox
                bbox = [bbox[0]-(prc_increase_aoi*int(bbox[0]/100)), bbox[1]-(prc_increase_aoi*int(bbox[1]/100)), bbox[2]+(prc_increase_aoi*int(bbox[0]/100)), bbox[3]+(prc_increase_aoi*int(bbox[3]/100))]
                B1.append(bbox)
                M1.append(p.major_axis_length)
                m1.append(p.minor_axis_length)
                A1.append(p.area)
                L1.append(t)
    return N1,F1,B1,M1,m1,A1,L1

##========
def write_csv(B1,F1,L1,A1,M1,m1,csv_file, flag=1):

    B = np.asarray(B1)
    ymin = B[:,0]
    xmin = B[:,1]
    ymax = B[:,2]
    xmax = B[:,3]
    w = xmax-xmin
    h= ymax-ymin

    F = []
    for f in F1:
        if 'post' in f:
            F.append(f.replace('labels','images').replace('_post_labelimage.png','.png'))
        elif 'pre' in f:
            F.append(f.replace('labels','images').replace('_pre_labelimage.png','.png'))

    C=[]
    if L1[0]=='building':
        C = [1 for c in L1]
    elif flag==2:
        for c in L1:
            if 'no' in c:
                C.append(0)
            elif 'dam' in c:
                C.append(1)
    else:
        for c in L1:
            if 'no' in c:
                C.append(0)
            elif 'minor' in c:
                C.append(1)
            elif 'major' in c:
                C.append(2)
            elif 'des' in c:
                C.append(3)
            elif 'un-' in c:
                C.append(4)

    d = {'filename':F,'width':w,'height':h,'class':C, 'xmin':xmin, 'xmax':xmax,
         'ymin':ymin, 'ymax':ymax, 'area':A1, 'mj_axis':M1, 'mn_axis':m1 }

    dataset = pd.DataFrame.from_dict(d)

    dataset.to_csv(csv_file)

##======================================================
def plot_samples(output_path, class_dict, flag=99):

    if flag==1:
        filenames = sorted(tf.io.gfile.glob(output_path+'buildings-*.tfrec'))
    elif flag==2:
        filenames = sorted(tf.io.gfile.glob(output_path+'nodamage*.tfrec'))
    else:
        filenames = sorted(tf.io.gfile.glob(output_path+'damage-*.tfrec'))

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)

    for i in dataset.take(25):
        name = i['image/filename'].numpy().decode()
        name = name.split(os.sep)[-1]

        image = tf.image.decode_jpeg(i['image'], channels=3)
        bbox = tf.numpy_function(np.array,[[i["objects/xmin"], i["objects/ymin"], i["objects/xmax"], i["objects/ymax"]]], tf.float32).numpy().T
        #print(len(bbox))

        if flag==1:
            ids = ['building' for id in i["objects/label"].numpy()]
        else:
            ids = []
            for id in i["objects/label"].numpy():
               ids.append(class_dict[id])

        fig =plt.figure(figsize=(16,16))
        plt.axis("off")
        plt.imshow(image)#, cmap=plt.cm.gray)
        ax = plt.gca()

        for box,id in zip(bbox,ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=[1 ,0, 0], linewidth=1)
            ax.add_patch(patch)
            ax.text(x1, y1, id, bbox={"facecolor": [1, 0, 0], "alpha": 0.4}, clip_box=ax.clipbox, clip_on=True, fontsize=8)
        #plt.show()
        if flag==1:
            plt.savefig('examples/'+name+'_bbox-buildings.png',dpi=200, bbox_inches='tight')
        elif flag==2:
            plt.savefig('examples/'+name+'_bbox-nodamage-damage.png',dpi=200, bbox_inches='tight')
        else:
            plt.savefig('examples/'+name+'_bbox-damage.png',dpi=200, bbox_inches='tight')
        plt.close('all')

##========================================== VARIABLES
ims_per_shard = 20

# classes = [0]
class_dict = {0: 'no-damage', 1: 'minor-damage', 2: 'major-damage', 3: 'destroyed'} #, 4: 'un-classified'}
#list of strings
# classes_string = [class_dict[i] for i in classes]

# Color codes for polygons
damage_dict = {
    "no-damage": (0, 255, 0),
    "minor-damage": (0, 0, 255),
    "major-damage": (255, 69, 0),
    "destroyed": (255, 0, 0),
}
    # "un-classified": (255, 255, 255)

##=============================================================================
features = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'objects/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'objects/ymin': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/xmax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/ymax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
}

cols = [damage_dict[k] for k in damage_dict]
types = [k for k in damage_dict]

prc_increase_aoi = 2

with tf.device('/cpu:0'):

    for storm in ['michael', 'florence', 'harvey', 'matthew']:

        imdir = '/media/marda/TWOTB1/xBD/hurricanes/images/'+storm
        lab_path = '/media/marda/TWOTB1/xBD/hurricanes/labels/'+storm
        csv_file = storm+'_bbox.csv'
        output_path = os.getcwd()+os.sep+'tfrecords/'+storm+'/bbox/'

        ##=============================================

        ## bbox: (min_row, min_col, max_row, max_col)
        ## major_axis_length, minor_axis_length, area, bbox_area

        N1,F1,B1,M1,m1,A1,L1 = get_bboxes(lab_path, cols, types, prc_increase_aoi)

        ###=============================================
        #create csv of filename,width,height,class,xmin,ymin,xmax,ymax
        write_csv(B1,F1,L1,A1,M1,m1,csv_file)

        ###=========================================
        write_records(csv_file,ims_per_shard,output_path)

        ###================ read back in to test
        plot_samples(output_path, class_dict)

        ##======================================
        ##just damage / no-damage
        index = np.where(np.array(L1) !='destroyed')[0]
        B = np.array(B1)[index].tolist()
        F = np.array(F1)[index].tolist()
        L = np.array(L1)[index].tolist()
        A = np.array(A1)[index].tolist()
        M = np.array(M1)[index].tolist()
        m = np.array(m1)[index].tolist()

        L = ['damage' if l.startswith('m') else 'no-damage' for l in L]

        csv_file = csv_file.replace('_bbox.csv','_bbox_nodamage-damage.csv')
        write_csv(B,F,L,A,M,m,csv_file,flag=2)
        del B1,F1,A1,M1,m1,L1

        write_records(csv_file,ims_per_shard,output_path,flag=2)
        plot_samples(output_path, class_dict, flag=2)

        ##======================================
        ##just buildings
        L = ['building' for l in L]

        csv_file = csv_file.replace('_bbox_nodamage-damage.csv','_bbox_buildings.csv')
        write_csv(B,F,L,A,M,m,csv_file)
        del B,F,A,M,m,L

        write_records(csv_file,ims_per_shard,output_path,flag=1)
        plot_samples(output_path, class_dict, flag=1)
