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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix #compute confusion matrix from vectors of observed and estimated labels


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# TARGET_SIZE = 1024
# BATCH_SIZE = 2
# MAX_EPOCHS = 100
# start_lr = 1e-6 #0.00001
# min_lr = start_lr
# max_lr = 1e-4
# rampup_epochs = 5
# sustain_epochs = 0
# exp_decay = .95

TARGET_SIZE = 1024
BATCH_SIZE = 5
MAX_EPOCHS = 100
start_lr = 1e-5
min_lr = start_lr
max_lr = 1e-3
rampup_epochs = 0
sustain_epochs = 5
exp_decay = .9

###############################################################
### MODEL FUNCTIONS
###############################################################
#-----------------------------------
def batchnorm_act(x):
    """
    batchnorm_act(x)
    This function applies batch normalization to a keras model layer, `x`, then a relu activation function
    INPUTS:
        * `z` : keras model layer (should be the output of a convolution or an input layer)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * batch normalized and relu-activated `x`
    """
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)

#-----------------------------------
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1)
    This function applies batch normalization to an input layer, then convolves with a 2D convol layer
    The two actions combined is called a convolutional block

    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`:input keras layer to be convolved by the block
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the batch normalized convolution
    """
    conv = batchnorm_act(x)
    return tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

#-----------------------------------
def bottleneck_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    bottleneck_block(x, filters, kernel_size=(3, 3), padding="same", strides=1)

    This function creates a bottleneck block layer, which is the addition of a convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between convolutional and bottleneck layers
    """
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([conv, bottleneck])

#-----------------------------------
def res_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    res_block(x, filters, kernel_size=(3, 3), padding="same", strides=1)

    This function creates a residual block layer, which is the addition of a residual convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([bottleneck, res])

#-----------------------------------
def upsamp_concat_block(x, xskip):
    """
    upsamp_concat_block(x, xskip)
    This function takes an input layer and creates a concatenation of an upsampled version and a residual or 'skip' connection
    INPUTS:
        * `xskip`: input keras layer (skip connection)
        * `x`: input keras layer
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    return tf.keras.layers.Concatenate()([u, xskip])

#-----------------------------------
def res_unet(sz, f, nclasses=1):
    """
    res_unet(sz, f, nclasses=1)
    This function creates a custom residual U-Net model for image segmentation
    INPUTS:
        * `sz`: [tuple] size of input image
        * `f`: [int] number of filters in the convolutional block
        * flag: [string] if 'binary', the model will expect 2D masks and uses sigmoid. If 'multiclass', the model will expect 3D masks and uses softmax
        * nclasses [int]: number of classes
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras model
    """
    inputs = tf.keras.layers.Input(sz)

    ## downsample
    e1 = bottleneck_block(inputs, f); f = int(f*2)
    e2 = res_block(e1, f, strides=2); f = int(f*2)
    e3 = res_block(e2, f, strides=2); f = int(f*2)
    e4 = res_block(e3, f, strides=2); f = int(f*2)
    _ = res_block(e4, f, strides=2)

    ## bottleneck
    b0 = conv_block(_, f, strides=1)
    _ = conv_block(b0, f, strides=1)

    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e3)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e2)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e1)
    _ = res_block(_, f)

    ## classify
    if nclasses==1:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="sigmoid")(_)
    else:
        outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="softmax")(_)

    #model creation
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

#-----------------------------------
def mean_iou(y_true, y_pred):
    """
    mean_iou(y_true, y_pred)
    This function computes the mean IoU between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * IoU score [tensor]
    """
    yt0 = y_true[:,:,:,0]
    yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

#-----------------------------------
def dice_coef(y_true, y_pred):
    """
    dice_coef(y_true, y_pred)

    This function computes the mean Dice coefficient between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice score [tensor]
    """
    smooth = 1.
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    dice_coef_loss(y_true, y_pred)

    This function computes the mean Dice loss (1 - Dice coefficient) between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice loss [tensor]
    """
    return 1.0 - dice_coef(y_true, y_pred)

#---------------------------------------------------
# learning rate function
def lrfn(epoch):
    """
    lrfn(epoch)
    This function creates a custom piecewise linear-exponential learning rate function for a custom learning rate scheduler. It is linear to a max, then exponentially decays

    * INPUTS: current `epoch` number
    * OPTIONAL INPUTS: None
    * GLOBAL INPUTS:`start_lr`, `min_lr`, `max_lr`, `rampup_epochs`, `sustain_epochs`, `exp_decay`
    * OUTPUTS:  the function lr with all arguments passed

    """
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)


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
    image = tf.image.decode_jpeg(bits)

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

#-----------------------------------
def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    #pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(y_pred, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]]) #pred_tensor

    # [b, classes]
    # count how many of each class are present in
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)


def dice_coef(y_true, y_pred):
    return 1-gen_dice(y_true, y_pred)


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


###############################################################
## VARIABLES
###############################################################

#==================================================================
# for storm in ['matthew', 'michael', 'florence', 'harvey', 'all']:

storm = 'all' #'harvey' #'florence' #'michael' #'matthew'

if storm is 'all':
    tfrecord_dir = []
    sample_data_path = []; sample_label_data_path = []
    for s in ['matthew', 'michael', 'florence', 'harvey']:
        # imdir.append('/media/marda/TWOTB1/xBD/hurricanes/images/'+s)
        # lab_path.append('/media/marda/TWOTB1/xBD/hurricanes/labels2D/'+s)
        tfrecord_dir.append('/media/marda/TWOTB1/xBD/hurricanes/tfrecords/'+s+'/imseg')
        sample_data_path.append('/media/marda/TWOTB1/xBD/hurricanes/samples/'+s)
        sample_label_data_path.append('/media/marda/TWOTB1/xBD/hurricanes/sample_labels/'+s)
    # weights = '/media/marda/TWOTB1/xBD/hurricanes/results/imseg/'+storm+'/'+storm+'_runet_weights_batch'+str(BATCH_SIZE)+'.h5'

else:
    # imdir = '/media/marda/TWOTB1/xBD/hurricanes/images/'+storm
    # lab_path =  '/media/marda/TWOTB1/xBD/hurricanes/labels2D/'+storm
    tfrecord_dir = '/media/marda/TWOTB1/xBD/hurricanes/tfrecords/'+storm+'/imseg'
    sample_data_path = '/media/marda/TWOTB1/xBD/hurricanes/samples/'+storm
    sample_label_data_path = '/media/marda/TWOTB1/xBD/hurricanes/sample_labels/'+storm

weights = '/media/marda/TWOTB1/xBD/hurricanes/results/imseg/'+storm+'/'+storm+'_runet_weights_batch'+str(BATCH_SIZE)+'.h5'
example_path = '/media/marda/TWOTB1/xBD/hurricanes/results/imseg/'+storm

#============================================================
hist_fig = weights.replace('.h5','_history.png')

test_samples_fig = hist_fig.replace('_history.png','_samples.png')

lr_fig = weights.replace('.h5','_lr.png')


patience = 10

ims_per_shard = 20

VALIDATION_SPLIT = 0.5

nclasses=6

###############################################################
## EXECUTION
###############################################################

#-------------------------------------------------
print('.....................................')
print('Reading files and making datasets ...')

if type(tfrecord_dir) is list:
    filenames = []
    for t in tfrecord_dir:
        filenames.append(sorted(tf.io.gfile.glob(t+os.sep+'*.tfrec')))

    filenames = np.hstack(filenames)

else:
    filenames = sorted(tf.io.gfile.glob(tfrecord_dir+os.sep+'*.tfrec'))

nb_images = ims_per_shard * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE

print(steps_per_epoch)
print(validation_steps)

train_ds = get_batched_dataset(training_filenames)
val_ds = get_batched_dataset(validation_filenames)


rng = [i for i in range(MAX_EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, [lrfn(x) for x in rng])
# plt.show()
plt.savefig(lr_fig, dpi=200, bbox_inches='tight')


print('.....................................')
print('Creating and compiling model ...')

model = res_unet((TARGET_SIZE, TARGET_SIZE, 3), BATCH_SIZE, nclasses)
#model.compile(optimizer = 'adam', loss = gen_dice, metrics = [mean_iou])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

# use multiclass Dice loss
# model3.compile(optimizer = 'adam', loss = multiclass_dice_coef_loss(), metrics = [mean_iou, multiclass_dice_coef])

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(weights, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

# models are sensitive to specification of learning rate. How do you decide? Answer: you don't. Use a learning rate scheduler

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

callbacks = [model_checkpoint, earlystop, lr_callback]

do_train = False# True

if do_train:
    print('.....................................')
    print('Training model ...')
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=val_ds, validation_steps=validation_steps,
                          callbacks=callbacks)

    # Plot training history
    n = len(history.history['val_loss'])

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(np.arange(1,n+1), history.history['mean_iou'], 'b', label='train accuracy')
    plt.plot(np.arange(1,n+1), history.history['val_mean_iou'], 'k', label='validation accuracy')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Mean IoU Coefficient', fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(122)
    plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
    plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(hist_fig, dpi=200, bbox_inches='tight')
    plt.close('all')
    K.clear_session()

else:
    model.load_weights(weights)

# ##########################################################
# ### evaluate
print('.....................................')
print('Evaluating model ...')
# testing
scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean IoU={mean_iou:0.4f}'.format(loss=scores[0], mean_iou=scores[1])) #mean_dice=scores[2]

#
# #mean loss = 0.16
# #mean iou = 0.95
# #mean dice =
#

for counter in range(1,10):
    for imgs,labs in val_ds.take(1):
        c=1
        for image,label in zip(imgs,labs):
            #print(image.shape)
            est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
            est_label = tf.argmax(est_label, axis=-1)
            print(np.unique(est_label))

            plt.subplot(2,3,c)
            plt.imshow(image, cmap='gray')
            plt.imshow(est_label, alpha=0.33, cmap=plt.cm.bwr)
            plt.axis('off')
            c +=1

            #plt.show()
            plt.savefig(example_path+'/model_output_example'+str(counter)+'.png', dpi=200, bbox_inches='tight')
            plt.close('all')

#
# ##########################################################
# ### predict
# print('.....................................')
# print('Using model for prediction on sample images ...')
#
# if type(sample_data_path) is list:
#     sample_filenames = []
#     for t in sample_data_path:
#         sample_filenames.append(sorted(tf.io.gfile.glob(t+os.sep+'*.png')))
#
#     sample_filenames = np.vstack(sample_filenames)
#
# else:
#     sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.png'))
#
# plt.figure(figsize=(16,16))
# imgs = []
# lbls = []
#
# for counter,f in enumerate(sample_filenames):
#     image = seg_file2tensor(f)/255
#     est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()#.astype(np.uint8)
#     est_label = tf.argmax(est_label, axis=-1)
#     print(np.unique(est_label))
#
#     plt.subplot(4,4,counter+1)
#     name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
#     plt.title(name, fontsize=10)
#     plt.imshow(image)
#     plt.imshow(est_label, alpha=0.5, cmap=plt.cm.bwr)#, vmin=0, vmax=6)
#
#     plt.axis('off')
#     imgs.append((image.numpy()*255).astype(np.uint8))
#     lbls.append(est_label)
#
# # plt.show()
# plt.savefig(test_samples_fig,
#             dpi=200, bbox_inches='tight')
# plt.close('all')
#
#
# print('.....................................')
# print('Using model for prediction on jpeg images in crf mode...')
#
# if type(sample_label_data_path) is list:
#     sample_label_filenames = []
#     for t in sample_label_data_path:
#         sample_label_filenames.append(sorted(tf.io.gfile.glob(t+os.sep+'*.png')))
#
#     sample_label_filenames = np.vstack(sample_label_filenames)
#
# else:
#     sample_label_filenames = sorted(tf.io.gfile.glob(sample_label_data_path+os.sep+'*.png'))
#
#
#
# obs = [np.array(seg_file2tensor(f), dtype=np.uint8).squeeze() for f in sample_label_filenames]
#
#
# for counter in range(len(obs)):
#     plt.subplot(221); plt.imshow(imgs[counter]); plt.imshow(obs[counter], alpha=0.5, cmap=plt.cm.bwr, vmin=0, vmax=255); plt.axis('off'); plt.title('Manual label', fontsize=6)
#     plt.savefig(os.getcwd()+'/example/gt-example'+str(counter)+'.png', dpi=600, bbox_inches='tight'); plt.close('all')
#
#
# cm = np.zeros((4,4))
# iou = []
# for k in range(len(obs)):
#     y = obs[k].copy()
#     ypred = lbls[k].copy()
#     #find unique values >0, but not in class set
#     u = np.setdiff1d(np.where(np.bincount(obs[k].flatten(), minlength=255)>0)[0], [62,63,64,65,66,126,127,128,129,130,189,190,191,192,193,253,254,255])
#     for item in u:
#         y[y==item]=4
#     y[y==62]=0; y[y==63]=0; y[y==64]=0; y[y==65]=0; y[y==66]=0;
#     y[y==128]=1; y[y==126]=1; y[y==127]=1; y[y==129]=1; y[y==130]=1;
#     y[y==189]=2; y[y==190]=2; y[y==191]=2; y[y==192]=2; y[y==193]=2;
#     y[y==255]=3; y[y==254]=3; y[y==253]=3;
#     y[y==4]=np.argmax(np.bincount(y.flatten(), minlength=5))
#
#     i = mean_iou_np(np.expand_dims(np.expand_dims(y,axis=0),axis=-1), np.expand_dims(np.expand_dims(ypred,axis=0),axis=-1))
#     iou.append(i)
#     cm += confusion_matrix(y.flatten(), ypred.flatten(), labels=np.arange(4))
#
#
# print('Mean IoU={mean_iou:0.3f}'.format(mean_iou=np.mean(iou)))

# # # mean iou = 0.89
#
# #
# # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# # cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
# #
# # plt.figure(figsize=(15,15))
# # plt.subplot(221)
# # ax = sns.heatmap(cm,
# #     annot=True,
# #     cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))
# #
# # ax.invert_yaxis()
# # ax.invert_xaxis()
# # # plt.show()
# #
# # CLASSES = ['no reef', 'reef']
# # tick_marks = np.arange(len(CLASSES))+.5
# # plt.xticks(tick_marks, CLASSES[::-1], rotation=45,fontsize=12)
# # plt.yticks(tick_marks, CLASSES[::-1],rotation=45, fontsize=12)
# #
# #
# # plt.subplot(223)
# # ax = sns.heatmap(cm2,
# #     annot=True,
# #     cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))
# #
# # ax.invert_yaxis()
# # ax.invert_xaxis()
# # # plt.show()
# #
# # plt.xticks(tick_marks, CLASSES[::-1], rotation=45,fontsize=12)
# # plt.yticks(tick_marks, CLASSES[::-1],rotation=45, fontsize=12)
# #
# # plt.savefig('results/OYSTERNET_cm_crf.png',
# #             dpi=200, bbox_inches='tight')
# # plt.close('all')
# #
# # print("Average true positive rate across %i classes: %.3f" % (len(CLASSES), np.mean(np.diag(cm))))
# # print("Average true positive rate across %i classes: %.3f" % (len(CLASSES), np.mean(np.diag(cm2))))
