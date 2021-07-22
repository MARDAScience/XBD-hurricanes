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

from glob import glob
from imageio import imwrite, imread
import os
import numpy as np

def write_flat_label(lab_path, cols):

    label = imread(lab_path)
    label2 = np.zeros(np.shape(label)[:2])
    counter=1
    for c in cols:
        mask = np.all(label==c,axis=2)
        label2[mask==1] = counter
        counter+=1

    imwrite(lab_path.replace('labels', 'labels2D'), label2.astype(np.uint8))
    #print(np.unique(label2.flatten()))


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

for storm in ['matthew', 'michael', 'florence', 'harvey']:

    lab_path =  '/media/marda/TWOTB1/xBD/hurricanes/labels/'+storm

    files = glob(lab_path+os.sep+'*.png')
    for lab_path in files:
        write_flat_label(lab_path, cols)
