import brian2 as b2 
import PIL 
import os
import sys
import numpy as np 
import pickle

def PoissonImages(imgfolder:str, presentation_time:float,
                  resize:tuple=None, contrast_multiplier:float=1,
                  imgmode:str='L',value_to_rate_coeff:float=0.25):
    """Load each images in imgfolder onto a Poisson neural population, 
    changing image every presentation_time ms

    Args:
        imgfolder (str): location of images
        presentation_time (float): presentation time of each image in ms
        resize (tuple, optional): Image size. If None no resize is performed.
                                  Defaults to None.
        contrast_multiplier (float, optional): Controls contrast of images. >1 
                                               is higher contrast. Defaults to 
                                               1.
        imgmode (str, optional): Color encoding (B&W, grayscale, RGB, etc..). 
                                 This is a parameter passed to 
                                 PIL.Image.convert. Defaults to grayscale.
        value_to_rate_coeff (float, optional): coeff to pass from pixel value 
                                               (0->255) to Poisson firing rate. 
                                               Defaults to 0.25.

    Returns:
        brian2.TimedArray: A TimedArray of firing rates representing images
        brian2.NeuronGroup: A Poisson neural population following these
                            firing rates
    """
    
    arrays = list()
    for filename in os.listdir(sys.path[0]+imgfolder):
        if filename.endswith(".jpg") or filename.endswith(".png") \
         or filename.endswith(".jpeg"):
            im = PIL.Image.open(os.path.join(sys.path[0]+imgfolder, filename))
            im = im.convert(imgmode)
            if resize is not None:
                im = im.resize(resize)
            if contrast_multiplier != 1:
                enhancer = PIL.ImageEnhance.Contrast(im)
                im = enhancer.enhance(contrast_multiplier)
            arrays.append(np.array(im).flatten())
        else:
            continue
    values = np.vstack(arrays)
    images = b2.TimedArray(values*b2.Hz*value_to_rate_coeff,
                           presentation_time*b2.ms)
    N = len(values[0])
    group = b2.NeuronGroup(N, 'rates : Hz', threshold='rand()<rates*dt')
    group.run_regularly('rates = images(t,i)', presentation_time*b2.ms)
    return images, group
    
def PoissonMNIST(presentation_time:float=1000, N:int=60000, resize:tuple=None,
                 contrast_multiplier:float=1, imgmode:str=None,
                 value_to_rate_coeff:float=0.25, name:str='PoissonMNIST'):
    """Load MNIST dataset onto a Poisson neural population changing image
    every presentation_time. Necessit of pickled file containing mnist_imgs the
    flatten pixel values of MNIST digits and mnist_labels the MNIST labels
    (that could be made more practical)

    Args:
        presentation_time (float, optional): presentation time of each image in 
                                             ms. Defaults to 1000.
        N (int, optional): Number of images to load. Defaults to 60000.
        resize (tuple, optional): Image size. If None no resize is performed.
                                  Defaults to None.
        contrast_multiplier (float, optional): Controls contrast of images. >1 
                                               is higher contrast. Defaults to 
                                               1.
        imgmode (str, optional): Color encoding (B&W, grayscale, RGB, etc..). 
                                 This is a parameter passed to
                                 PIL.Image.convert. Defaults to grayscale.
                                 
        value_to_rate_coeff (float, optional): coeff to pass from pixel value 
                                               (0->255) to Poisson firing rate. 
                                               Defaults to 0.25.
        name (str, optional): Name of the neural population. Defaults to 
                              'PoissonMNIST'.

    Returns:
        brian2.TimedArray: A TimedArray of firing rates representing MNIST
                           digits
        brian2.NeuronGroup: A Poisson neural population following these
                            firing rates
        list of int: Associated MNIST labels
        float * brian2.ms: Total simulation time for each image to be presented
    """
    
    f = open(sys.path[0]+"/mnist", "rb")
    mnist_imgs = pickle.load(f)[:N,:]
    mnist_labels = pickle.load(f)[:N]
    f.close()
    p = np.random.permutation(len(mnist_imgs))
    mnist_imgs, mnist_labels = mnist_imgs[p], mnist_labels[p]
    timed_spikes = b2.TimedArray(value_to_rate_coeff*mnist_imgs*b2.Hz,
                                 presentation_time*b2.ms)
    M = len(mnist_imgs[0])
    group = b2.NeuronGroup(M, 'rates : Hz', threshold='rand()<rates*dt',
                           name=name)
    group.run_regularly('rates = images(t,i)', presentation_time*b2.ms) 
    return timed_spikes, group, mnist_labels, N * presentation_time*b2.ms