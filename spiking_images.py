import brian2 as b2 
import PIL 
import os
import sys
import numpy as np 
import pickle

def PoissonImages(imgfolder, presentation_time, resize=None, contrast_multiplier=1, imgmode=None):
    arrays = list()
    for filename in os.listdir(sys.path[0]+imgfolder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            im = PIL.Image.open(os.path.join(sys.path[0]+imgfolder, filename))
            im = im.convert(imgmode) if imgmode is not None else im.convert('L')
            if resize is not None:
                im = im.resize(resize)
            if contrast_multiplier != 1:
                enhancer = PIL.ImageEnhance.Contrast(im)
                im = enhancer.enhance(contrast_multiplier)
            arrays.append(np.array(im).flatten())
        else:
            continue
    values = np.vstack(arrays)
    images = b2.TimedArray(values*b2.Hz/4, presentation_time)
    N = len(values[0])
    group = b2.NeuronGroup(N, 'rates : Hz', threshold='rand()<rates*dt')
    group.run_regularly('rates = images(t,i)', presentation_time)
    return images, group
    
def PoissonMNIST(presentation_time=1000*b2.ms, N=60000, resize=None, contrast_multiplier=1, imgmode=None, value_to_rate_coeff=0.25, name=''):
    f = open(sys.path[0]+"/mnist", "rb")
    mnist_imgs = pickle.load(f)[:N,:]
    mnist_labels = pickle.load(f)[:N]
    f.close()
    p = np.random.permutation(len(mnist_imgs))
    mnist_imgs, mnist_labels = mnist_imgs[p], mnist_labels[p]
    timed_spikes = b2.TimedArray(value_to_rate_coeff*mnist_imgs*b2.Hz, presentation_time)
    M = len(mnist_imgs[0])
    group = b2.NeuronGroup(M, 'rates : Hz', threshold='rand()<rates*dt', name=name)
    group.run_regularly('rates = images(t,i)', presentation_time) 
    return timed_spikes, group, mnist_labels, N * presentation_time