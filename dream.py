# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Deep Dreams (with Caffe)
# 
# This notebook demonstrates how to use the [Caffe](http://caffe.berkeleyvision.org/) neural network framework to produce "dream" visuals shown in the [Google Research blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html).
# 
# It'll be interesting to see what imagery people are able to generate using the described technique. If you post images to Google+, Facebook, or Twitter, be sure to tag them with **#deepdream** so other researchers can check them out too.
# 
# ##Dependencies
# This notebook is designed to have as few dependencies as possible:
# * Standard Python scientific stack: [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [PIL](http://www.pythonware.com/products/pil/), [IPython](http://ipython.org/). Those libraries can also be installed as a part of one of the scientific packages for Python, such as [Anaconda](http://continuum.io/downloads) or [Canopy](https://store.enthought.com/).
# * [Caffe](http://caffe.berkeleyvision.org/) deep learning framework ([installation instructions](http://caffe.berkeleyvision.org/installation.html)).
# * Google [protobuf](https://developers.google.com/protocol-buffers/) library that is used for Caffe model manipulation.

# <codecell>

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import os, fnmatch

import caffe

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

# <markdowncell>

# ## Loading DNN model
# In this notebook we are going to use a [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) model trained on [ImageNet](http://www.image-net.org/) dataset.
# Feel free to experiment with other models from Caffe [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). One particularly interesting [model](http://places.csail.mit.edu/downloadCNN.html) was trained in [MIT Places](http://places.csail.mit.edu/) dataset. It produced many visuals from the [original blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html).

# <codecell>

model_path = '/home/ubuntu/caffe/models/googlenet_places205/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'googlenet_places205.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# <headingcell level=2>

#  Producing dreams

# <markdowncell>

# Making the "dream" images is very simple. Essentially it is just a gradient ascent process that tries to maximize the L2 norm of activations of a particular DNN layer. Here are a few simple tricks that we found useful for getting good images:
# * offset image by a random jitter
# * normalize the magnitude of gradient ascent steps
# * apply ascent across multiple scales (octaves)
# 
# First we implement a basic gradient ascent step function, applying the first two tricks:

# <codecell>

def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)    

# <markdowncell>

# Next we implement an ascent through different scales. We call these scales "octaves".

# <codecell>

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            clear_output()
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

# <markdowncell>

# Now we are ready to let the neural network reveal its dreams! Let's take a [cloud image](https://commons.wikimedia.org/wiki/File:Appearance_of_sky_for_weather_forecast,_Dhaka,_Bangladesh.JPG) as a starting point:

# <codecell>

#img = np.float32(PIL.Image.open('selfie.jpg'))

# <markdowncell>

# Running the next code cell starts the detail generation process. You may see how new patterns start to form, iteration by iteration, octave by octave.

# <codecell>

layer = 'inception_4b/5x5_reduce'

stream = 0
if (stream == 1):
    path = 'dreaming/' + os.listdir('dreaming')[0]
    img = np.float32(PIL.Image.open(path))
    frame = img
    frame_i = 0

    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    for i in xrange(20):
        frame = deepdream(net, frame, end=layer)
        PIL.Image.fromarray(np.uint8(frame)).save("frames/%04d.jpg"%frame_i)
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        frame_i += 1
elif (stream == 0):
    for imgfile in os.listdir('dreaming'):
        path = 'dreaming/' + imgfile
        img = np.float32(PIL.Image.open(path))
        dream = deepdream(net, img, end=layer)
        newpath = 'dreaming/' + os.path.splitext(imgfile)[0] + '-' + os.path.splitext(os.path.basename(param_fn))[0] + '-' + layer.replace('/', '_') + '.jpg'
        PIL.Image.fromarray(np.uint8(dream)).save(newpath)
        print newpath,' has completed successfully.\n\n'

        c = 1
        while c < 6:
            dream = deepdream(net, dream, end=layer)
            newpath = 'dreaming/' + os.path.splitext(imgfile)[0] + '-' + str(c) + '.jpg'
            PIL.Image.fromarray(np.uint8(dream)).save(newpath)
            print newpath,' has completed successfully.\n\n'
            c += 1

# <markdowncell>
