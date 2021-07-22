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
# from sklearn.metrics import confusion_matrix #compute confusion matrix from vectors of observed and estimated labels

from collections import OrderedDict
from random import shuffle


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


BATCH_SIZE = 4 #2
MAX_EPOCHS = 200

nclasses=4 #1


# # ## scratch learning rate curve
start_lr = 1e-07
min_lr = start_lr
max_lr = 1e-04
rampup_epochs = 15
sustain_epochs = 5
exp_decay = .8


# ## transfer learning rate curve
# start_lr = 1e-06
# min_lr = start_lr
# max_lr = 1e-04
# rampup_epochs = 10
# sustain_epochs = 5
# exp_decay = .8



###############################################################
### MODEL FUNCTIONS
###############################################################
#----------------------------------------------
def get_inference_model(threshold, model): #, num_classes):
    """
    get_inference_model(threshold, model)
    This function creates an inference model consisting of an input layer for an image
    the model predictions, decoded detections, then finally a mapping from image to detections
    In effect it is a model nested in another model
    INPUTS:
        * threshold [float], the detecton probability beyond which we are confident of
        * model [keras model], trained object detection model
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay
    OUTPUTS:  keras model for detections on images
    """
    # ANY size input
    image = tf.keras.Input(shape=[None, None, 3], name="image")

    predictions = model(image, training=False)

    detections = DecodePredictions(confidence_threshold=threshold)(image, predictions)

    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    return inference_model


def visualize_detections(
    image, boxes, classes, scores, counter, str_prefix, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """
    visualize_detections(image, boxes, classes, scores, counter, str_prefix, figsize=(7, 7), linewidth=1, color=[0, 0, 1])
    This function allows for visualization of imagery and bounding boxes

    INPUTS:
        * images [ndarray]: batch of images
        * boxes [ndarray]: batch of bounding boxes per image
        * classes [list]: class strings
        * scores [list]: prediction scores
        * str_prefix [string]: filename prefix
    OPTIONAL INPUTS:
      * figsize=(7, 7)
      * linewidth=1
      * color=[0, 0, 1]
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """
    image = np.array(image, dtype=np.uint8)
    fig =plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.savefig(str_prefix+str(counter)+'.png', dpi=200, bbox_inches='tight')
    plt.close('all')

#-----------------------------------
def file2tensor(f):
    """
    file2tensor(f)
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained mobilenet or vgg model
    (the imagery is standardized depedning on target model framework)
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS:
        * model = {'mobilenet' | 'vgg'}
    OUTPUTS:
        * image [tensor array]: unstandardized image
        * im [tensor array]: standardized image
    GLOBAL INPUTS: TARGET_SIZE
    """
    bits = tf.io.read_file(f)
    image = tf.image.decode_png(bits)

    return image

def lrfn(epoch):
    """
    lrfn(epoch)
    This function creates a custom piecewise linear-exponential learning rate function
    for a custom learning rate scheduler. It is linear to a max, then exponentially decays
    INPUTS: current epoch number
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay
    OUTPUTS:  the function lr with all arguments passed
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

def prepare_image(image):
    """
    prepare_image(image)
    ""
    This function resizes and pads an image, and rescales for resnet
    INPUTS:
        * image [tensor array]
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
    GLOBAL INPUTS: None
    """
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


def compute_iou(boxes1, boxes2):
    """
    compute_iou(boxes1, boxes2)
    This function computes pairwise IOU matrix for given two sets of boxes
    INPUTS:
        * boxes1: A tensor with shape `(N, 4)` representing bounding boxes
          where each box is of the format `[x, y, width, height]`.
        * boxes2: A tensor with shape `(M, 4)` representing bounding boxes
          where each box is of the format `[x, y, width, height]`.
    OPTIONAL INPUTS: None
    OUTPUTS:
        *  pairwise IOU matrix with shape `(N, M)`, where the value at ith row
           jth column holds the IOU between ith box and jth box from
           boxes1 and boxes2 respectively.
    GLOBAL INPUTS: None
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


# ## Implementing Anchor generator
# Anchor boxes are fixed sized boxes that the model uses to predict the bounding
# box for an object. It does this by regressing the offset between the location
# of the object's center and the center of an anchor box, and then uses the width
# and height of the anchor box to predict a relative scale of the object. In the
# case of RetinaNet, each location on a given feature map has nine anchor boxes
# (at three scales and three ratios).

class AnchorBox:
    """
    "AnchorBox"
    ## Code from https://keras.io/examples/vision/retinanet/
    Generates anchor boxes.
    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.
    INPUTS:
      * aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      * scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      * num_anchors: The number of anchor boxes at each location on feature map
      * areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      * strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * anchor boxes for all the feature maps, stacked as a single tensor with shape
        `(total_anchors, 4)`, when AnchorBox._get_anchors() is called
    GLOBAL INPUTS: None
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """
        "_get_anchors"
        ## Code from https://keras.io/examples/vision/retinanet/
        Generates anchor boxes for a given feature map size and level
        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.
        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """
        "get_anchors"
        ## Code from https://keras.io/examples/vision/retinanet/
        Generates anchor boxes for all the feature maps of the feature pyramid.
        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.
        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)



def get_backbone(noise_stdev=0.1):
    """
    get_backbone()
    ## Code from https://keras.io/examples/vision/retinanet/
    ""
    This function Builds ResNet50 with pre-trained imagenet weights
    INPUTS: None
    OPTIONAL INPUTS: None
    OUTPUTS:
        * keras Model
    GLOBAL INPUTS: BATCH_SIZE
    """
    backbone = tf.keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    # return tf.keras.Model(
    #     inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    # )
    return tf.keras.Model(
        inputs=[backbone.inputs], outputs=[tf.keras.layers.GaussianNoise(noise_stdev)(c3_output), tf.keras.layers.GaussianNoise(noise_stdev)(c4_output), tf.keras.layers.GaussianNoise(noise_stdev)(c5_output)]
    )


"""
## Building Feature Pyramid Network as a custom layer
"""


class FeaturePyramid(tf.keras.layers.Layer):
    """
    "FeaturePyramid"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class builds the Feature Pyramid with the feature maps from the backbone.
    INPUTS:
      * num_classes: Number of classes in the dataset.
      * backbone: The backbone to build the feature pyramid from. Currently supports ResNet50 only (the output of get_backbone())
    OPTIONAL INPUTS: None
    OUTPUTS:
        * the 5-feature pyramids (feature maps) at strides `[8, 16, 32, 64, 128]`
    GLOBAL INPUTS: None
    """

    def __init__(self, noise_stdev=0.1, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone(noise_stdev)
        self.conv_c3_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = tf.keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


def build_head(output_filters, bias_init):
    """
    "build_head(output_filters, bias_init)"
    ## Code from https://keras.io/examples/vision/retinanet/
    This function builds the class/box predictions head.
    INPUTS:
        * output_filters: Number of convolution filters in the final layer.
        * bias_init: Bias Initializer for the final convolution layer.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * a keras sequential model representing either the classification
          or the box regression head depending on `output_filters`.
    GLOBAL INPUTS: None
    """
    head = tf.keras.Sequential([tf.keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(tf.keras.layers.ReLU())
    head.add(
        tf.keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head


"""
## Building RetinaNet using a subclassed model
"""


class RetinaNet(tf.keras.Model):
    """
    "RetinaNet"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class returns a subclassed Keras model implementing the RetinaNet architecture.
    INPUTS:
        * num_classes: Number of classes in the dataset.
        * backbone: The backbone to build the feature pyramid from. Supports ResNet50 only.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """

    def __init__(self, num_classes, noise_stdev, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(noise_stdev, backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


"""
## Implementing a custom layer to decode predictions
"""


class DecodePredictions(tf.keras.layers.Layer):
    """
    "DecodePredictions"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class creates a Keras layer that decodes predictions of the RetinaNet model.
    INPUTS:
        * num_classes: Number of classes in the dataset
        * confidence_threshold: Minimum class probability, below which detections
          are pruned.
        * nms_iou_threshold: IOU threshold for the NMS operation
        * max_detections_per_class: Maximum number of detections to retain per class.
        * max_detections: Maximum number of detections to retain across all classes.
        * box_variance: The scaling factors used to scale the bounding box predictions.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * a keras layer to decode predictions
    GLOBAL INPUTS: None
    """

    def __init__(
        self,
        num_classes=1,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


###############################################################
## MODEL TRAINING
###############################################################

class RetinaNetBoxLoss(tf.losses.Loss):
    """
    "RetinaNetBoxLoss"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class implements smooth L1 loss
    INPUTS:
        * y_true [tensor]: label observations
        * y_pred [tensor]: label estimates
    OPTIONAL INPUTS: None
    OUTPUTS:
        * loss [tensor]
    GLOBAL INPUTS: None
    """

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """
    "RetinaNetClassificationLoss"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class implements Focal loss
    INPUTS:
        * y_true [tensor]: label observations
        * y_pred [tensor]: label estimates
    OPTIONAL INPUTS: None
    OUTPUTS:
        * loss [tensor]
    GLOBAL INPUTS: None
    """

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """
    "RetinaNetLoss"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class is a wrapper to sum RetinaNetClassificationLoss and RetinaNetClassificationLoss outputs
    INPUTS:
        * y_true [tensor]: label observations
        * y_pred [tensor]: label estimates
    OPTIONAL INPUTS: None
    OUTPUTS:
        * loss [tensor]
    GLOBAL INPUTS: None
    """

    def __init__(self, num_classes=1, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss



class LabelEncoderCoco:
    """
    LabelEncoderCoco()
    Transforms the raw labels into targets for training.
    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.
    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.
        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.
        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.
        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()

def convert_to_corners(boxes):
    """
    convert_to_corners(boxes)
    Changes the box format to corner coordinates
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


#----------------------------------------------
def prepare_secoora_datasets_for_training(train_filenames, val_filenames):
    """
    prepare_secoora_datasets_for_training(train_filenames, val_filenames):
    This funcion prepares train and validation datasets  by extracting features (images, bounding boxes, and class labels)
    then map to preprocess_secoora_data, then apply prefetch, padded batch and label encoder
    INPUTS:
        * train_filenames [string]: tfrecord filenames for training
        * val_filenames [string]: tfrecord filenames for validation
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """

    features = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'objects/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'objects/ymin': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/xmax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/ymax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
    }


    def _parse_function(example_proto):
      # Parse the input `tf.train.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, features)

    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(_parse_function)

    train_dataset = train_dataset.map(preprocess_secoora_data, num_parallel_calls=AUTO)

    shapes = (tf.TensorShape([None,None,3]),tf.TensorShape([None,4]),tf.TensorShape([None,]))

    # this is necessary because there are unequal numbers of labels in every image
    train_dataset = train_dataset.padded_batch(
        batch_size = BATCH_SIZE, drop_remainder=True, padding_values=(0.0, 1e-8, -1), padded_shapes=shapes,
    )

    label_encoder = LabelEncoderCoco()

    # train_dataset = train_dataset.shuffle(8 * BATCH_SIZE)
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=AUTO
    )

    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(AUTO)

    val_dataset = tf.data.TFRecordDataset(val_filenames)
    val_dataset = val_dataset.map(_parse_function)
    val_dataset = val_dataset.map(preprocess_secoora_data, num_parallel_calls=AUTO)

    val_dataset = val_dataset.padded_batch(
        batch_size = BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True, padded_shapes=shapes,
    )

    val_dataset = val_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=AUTO
    )
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(AUTO)

    return train_dataset, val_dataset



def preprocess_secoora_data(example):
    """
    preprocess_secoora_data(example)
    ""
    This function
    INPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    #image = tf.image.per_image_standardization(image)

    bbox = tf.numpy_function(np.array,[[example["objects/xmin"], example["objects/ymin"], example["objects/xmax"], example["objects/ymax"]]], tf.float32)
    bbox = tf.transpose(bbox)

    class_id = tf.cast(example["objects/label"], dtype=tf.int32)

    # image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, ratio = resize_and_pad_image(image)

    bbox3 = tf.stack(
        [
            bbox[:, 0] * ratio,
            bbox[:, 1] * ratio,
            bbox[:, 2] * ratio,
            bbox[:, 3] * ratio,
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(tf.cast(bbox3, tf.float32))

    return image, bbox, class_id



def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """
    resize_and_pad_image(image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0)
    Resizes and pads image while preserving aspect ratio.
    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`
    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.
    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio

def convert_to_xywh(boxes):
    """
    convert_to_xywh(boxes)
    Changes the box format to center, width and height.
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )

def int2str(class_dict, classes):
    return [class_dict[i] for i in classes]

def int2name(storms, item):
    return storms[item]

###############################################################
## VARIABLES
###############################################################

patience = 5

ims_per_shard = 20

VALIDATION_SPLIT = 0.9 #7

nclasses=4 #5

class_dict = {0: 'no-damage', 1: 'minor-damage', 2: 'major-damage', 3: 'destroyed'} #, 4: 'un-classified'}
storms = ['matthew', 'michael', 'florence', 'harvey']

threshold = 0.5

do_train =  True

transfer_learn = False #True

noise_stdev = 0.3
gamma = 5.0 #4.0
alpha = 0.025

transfer_weights = os.path.join('/media/marda/TWOTB1/xBD/hurricanes/results/bbox/florence', "damage-scratch_weights")

#==================================================================
for storm in ['all']: #['michael', 'florence', 'harvey', 'matthew', 'all']:

    if storm is 'all':
        tfrecord_dir = []
        sample_data_path = []; sample_label_data_path = []
        for s in ['matthew', 'michael', 'florence', 'harvey']:
            tfrecord_dir.append('/media/marda/TWOTB1/xBD/hurricanes/tfrecords/'+s+'/bbox')
            sample_data_path.append('/media/marda/TWOTB1/xBD/hurricanes/samples/'+s)
            sample_label_data_path.append('/media/marda/TWOTB1/xBD/hurricanes/sample_labels/'+s)
    else:
        tfrecord_dir = '/media/marda/TWOTB1/xBD/hurricanes/tfrecords/'+storm+'/bbox'
        sample_data_path = []; sample_label_data_path = []
        for s in ['matthew', 'michael', 'florence', 'harvey']:
            sample_data_path.append('/media/marda/TWOTB1/xBD/hurricanes/samples/'+s)
            sample_label_data_path.append('/media/marda/TWOTB1/xBD/hurricanes/sample_labels/'+s)

    if transfer_learn is True:
        weights = '/media/marda/TWOTB1/xBD/hurricanes/results/bbox/'+storm+'/damage-transfer_weights'
    else:
        weights = '/media/marda/TWOTB1/xBD/hurricanes/results/bbox/'+storm+'/damage-scratch_weights'

    weights_path = '/media/marda/TWOTB1/xBD/hurricanes/results/bbox/'+storm

    #============================================================
    hist_fig = weights+'_history.png'

    test_samples_fig = hist_fig.replace('_history.png','_samples.png')

    lr_fig = weights.replace('.h5','_lr.png')

    ###############################################################
    ## EXECUTION
    ###############################################################

    #-------------------------------------------------
    print('.....................................')
    print('Reading files and making datasets ...')

    if type(tfrecord_dir) is list:
        filenames = []
        for t in tfrecord_dir:
            filenames.append(sorted(tf.io.gfile.glob(t+os.sep+'damage-*.tfrec')))

        filenames = np.hstack(filenames)

    else:
        filenames = sorted(tf.io.gfile.glob(tfrecord_dir+os.sep+'damage-*.tfrec'))


    shuffle(filenames)

    nb_images = ims_per_shard * len(filenames)
    print(nb_images)

    split = int(len(filenames) * VALIDATION_SPLIT)

    training_filenames = filenames[split:]
    validation_filenames = filenames[:split]

    validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
    steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE

    print(steps_per_epoch)
    print(validation_steps)

    val_dataset, train_dataset = prepare_secoora_datasets_for_training(training_filenames, validation_filenames)

    """
    ## LR
    """

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

    rng = [i for i in range(MAX_EPOCHS)]
    y = [lrfn(x) for x in rng]
    plt.plot(rng, [lrfn(x) for x in rng])
    # plt.show()
    if transfer_learn is True:
        plt.savefig(os.getcwd()+os.sep+'results/learnratesched_transfer_bbox.png', dpi=200, bbox_inches='tight')
    else:
        plt.savefig(os.getcwd()+os.sep+'results/learnratesched_scratch_bbox.png', dpi=200, bbox_inches='tight')


    """
    ## Initializing model
    """
    print('.....................................')
    print('Creating and compiling model ...')

    resnet50_backbone = get_backbone(noise_stdev)
    loss_fn = RetinaNetLoss(nclasses,  alpha=alpha, gamma=gamma) # num_classes=80, alpha=0.25, gamma=2.0)


    # gamma controls degree of focus on hard, misclassified examples. "focusing parameter" or the modulating factor reduces the loss contri-bution from easy examples and extends the range in whichan example receives low loss.
    # alpha is the "alpha balance"
    # focal loss = -alpha(1-p)^gamma log(p)

    model = RetinaNet(nclasses, noise_stdev, resnet50_backbone)



    """
    ## Setting up callbacks, and compiling model
    """

    earlystop = EarlyStopping(monitor="val_loss",
                                  mode="min", patience=patience)

    if transfer_learn is True:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(weights_path, "damage-transfer_weights"), # + "_epoch_{epoch}"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
    else:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(weights_path, "damage-scratch_weights"), # + "_epoch_{epoch}"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )

    optimizer = tf.optimizers.Adam() #SGD(momentum=0.9)
    # optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    # no loading of weights - training from scratch
    callbacks = [model_checkpoint, earlystop, lr_callback]

    """
    ## Training the model
    """

    if transfer_learn is True:
        model.load_weights(transfer_weights)

    if do_train is True:

        print('.....................................')
        print('Training model ...')
        history = model.fit(
            train_dataset, validation_data=val_dataset, epochs=MAX_EPOCHS, callbacks=callbacks)

        # history.history.keys()

        K.clear_session()

        n = len(history.history['loss'])

        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
        plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
        plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
        plt.legend(fontsize=10)

        # plt.show()
        plt.savefig(hist_fig, dpi=200, bbox_inches='tight')
        plt.close('all')

    else:
        model.load_weights(weights)

    # ##########################################################
    # ### evaluate
    print('.....................................')
    print('Evaluating model ...')
    # testing
    scores = model.evaluate(val_dataset, steps=validation_steps)
    print(scores)

    #harvey = 1.67
    #matthew = 1.24
    #michael = 1 .54
    #florence = 0 .8
    #all = 1.36

    # print('loss={loss:0.4f}, Mean IoU={mean_iou:0.4f}'.format(loss=scores[0], mean_iou=scores[1]))
    # #mean loss = 2.017 / 1.93


    inference_model = get_inference_model(threshold, model)

    for item in range(len(sample_data_path)):

        sample_filenames = sorted(tf.io.gfile.glob(sample_data_path[item]+os.sep+'*.png'))

        for counter,f in enumerate(sample_filenames):
            image = file2tensor(f)

            image = tf.cast(image, dtype=tf.float32)
            input_image, ratio = prepare_image(image)
            detections = inference_model.predict(input_image)
            num_detections = detections.valid_detections[0]

            boxes = detections.nmsed_boxes[0][:num_detections] / ratio
            scores = detections.nmsed_scores[0][:num_detections]

            classes = int2str(class_dict, detections.nmsed_classes[0][:num_detections])
            print(classes)

            # visualize_detections(image, boxes, classes,scores)
            visualize_detections(image, boxes, classes,scores, counter, weights_path+'/'+int2name(storms,item)+'_obrecog_example', figsize=(16, 16), linewidth=1, color=[0, 1, 0])
            plt.close('all')
