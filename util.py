import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import imageio.v2 as imageio
import math

def read_image(img_path='data/78004.jpg'):
    '''
    this function reads img path and returns the image
    Args: 
        img_path (str): path to the image file
    Returns:
        imagio image object    
    '''
    return imageio.imread(img_path).astype('int32')

def show(img):
    '''
    this function is used to display the image
    Args:
        img : image object
    '''
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

def show_beforeafter(I, final_image):
    '''
    disply the before and after images side by side
    Args:
        I: original image
        final_image: image after applying Canny Edge Detection
    '''
    plt.figure()
    f, axarr = plt.subplots(1,2) 

    axarr[0].imshow(I, cmap = plt.get_cmap('gray'))
    axarr[1].imshow(final_image, cmap = plt.get_cmap('gray'))

    plt.show()

def show_sidebyside(img_list, title_list=None):
    '''
    displays all the images in the list side by side
    Args:
        img_list (list): list of images
    '''
    plt.figure()
    f, ax = plt.subplots(1,len(img_list), figsize=(10,7)) 
    for i, img in enumerate(img_list):
        if title_list is not None:
            ax[i].set_title(title_list[i])
        ax[i].imshow(img, cmap = plt.get_cmap('gray'))

    plt.show() 

def gaussmask(sigma, kernelsize=[3,3]):
    '''
    produces a 1D gaussian mask/kernel of given standard deviation (sigma) and kernel size
    Args:
        sigma (float): standard deviation of the gaussian function
        kernelsize (array/list/tuple): [row, column] size of the kernel 
    Returns:
        Normalised numpy 1D array gaussion function of size (kernelsize) 
    '''
    mid = kernelsize[0] 
    g = np.array([(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]) # gaussian function
    return g/g.sum() # return the normalised gaussian function

def gaussblur(I, sigma=1.4, kernelsize=[3, 3]):
    G = gaussmask(sigma, [3,3])[np.newaxis] # create a 1D gaussian mask G to convolve with I
    return ndimage.convolve(I, G * G.T)

def hypot(Ix, Iy):
    '''
    computes the magnitute of each element of Ix and Iy
    Args:
        Ix (np.array): one of the 2D arrays 
        Iy (np.array): one of the 2D arrays 
    Returns:
        mag (np.array): an array consisting the magnitute each x and y component
    '''
    mag = np.zeros(Ix.shape) # create an empty array to store the magnitute values
    for i in range(Ix.shape[0]): # iterate over all the rows
        for j in range(Ix.shape[1]): # iterate over all the columns
            mag[i][j] = math.sqrt(Ix[i][j]**2 + Iy[i][j]**2) # compute the magnitute of the x and y element at (i,j) 
    return mag

def NonMaxSup(I_mag, I_orient):
    '''
    This function computes Non-Max Supression of the magnitudes of the each pixel based on their orientation
    Args:
        I_mag (np.array): an array consisting the magnitute each pixel
        I_orient (np.array): an array consisting the orientation at each pixel
    Return:
        an array consisting the non-max suppressed magnitude values at eaxh pixel
    '''
    NMS = np.zeros(I_mag.shape) # create an empty array to store the non-max suppressed magnitude values later
    for i in range(1, int(I_mag.shape[0]) - 1): # iterate over the rows
        for j in range(1, int(I_mag.shape[1]) - 1): # iterate over the columns
            # if the orientation at the pixel (i,j) is between -22.5 and 22.5 or between -157.5 and 157.5
            if((I_orient[i,j] >= -22.5 and I_orient[i,j] <= 22.5) or (I_orient[i,j] <= -157.5 and I_orient[i,j] >= 157.5)):
                # if the magnitude at the pixel (i,j) is greater than both it's adjacent pixel values at (i, j+1) and (i, j-1)
                if((I_mag[i,j] > I_mag[i,j+1]) and (I_mag[i,j] > I_mag[i,j-1])):
                    # save the magnitude value at pixel (i, j) since it's the max value in that region 
                    NMS[i,j] = I_mag[i,j]
                else: # all the other values are non-max suppressed
                    NMS[i,j] = 0
            # if the orientation at the pixel (i,j) is between 22.5 and 67.5 or between -112.5 and -157.5                    
            if((I_orient[i,j] >= 22.5 and I_orient[i,j] <= 67.5) or (I_orient[i,j] <= -112.5 and I_orient[i,j] >= -157.5)):
                # if the magnitude at the pixel (i,j) is greater than both it's adjacent pixel values at (i+1, j+1) and (i-1, j-1)
                if((I_mag[i,j] > I_mag[i+1,j+1]) and (I_mag[i,j] > I_mag[i-1,j-1])):
                    # save the magnitude value at pixel (i, j) since it's the max value in that region 
                    NMS[i,j] = I_mag[i,j]
                else: # all the other values are non-max suppressed
                    NMS[i,j] = 0
            # if the orientation at the pixel (i,j) is between 67.5 and 112.5 or between -67.5 and -112.5                                        
            if((I_orient[i,j] >= 67.5 and I_orient[i,j] <= 112.5) or (I_orient[i,j] <= -67.5 and I_orient[i,j] >= -112.5)):
                # if the magnitude at the pixel (i,j) is greater than both it's adjacent pixel values at (i+1, j) and (i-1, j)
                if((I_mag[i,j] > I_mag[i+1,j]) and (I_mag[i,j] > I_mag[i-1,j])):
                    # save the magnitude value at pixel (i, j) since it's the max value in that region 
                    NMS[i,j] = I_mag[i,j]
                else:  # all the other values are non-max suppressed
                    NMS[i,j] = 0
            # if the orientation at the pixel (i,j) is between 112.5 and 157.5 or between -22.5 and -67.5 
            if((I_orient[i,j] >= 112.5 and I_orient[i,j] <= 157.5) or (I_orient[i,j] <= -22.5 and I_orient[i,j] >= -67.5)):
                # if the magnitude at the pixel (i,j) is greater than both it's adjacent pixel values at (i+1, j-1) and (i-1, j+1)
                if((I_mag[i,j] > I_mag[i+1,j-1]) and (I_mag[i,j] > I_mag[i-1,j+1])):
                    # save the magnitude value at pixel (i, j) since it's the max value in that region 
                    NMS[i,j] = I_mag[i,j]
                else:  # all the other values are non-max suppressed
                    NMS[i,j] = 0

    return NMS

def DoThreshHyst(img, HTR=0.32, LTR=0.30):
    '''
    performs Double Threshold Hysterisis by recursing through every strong edge and find all connected weak edges
    if the pixel's intensity is less than the low threshold value it is supressed
    if it's between low and high threshold then we check if any neighbouring pixel has intensity above the high threshold otherwise 
        if yes, we consider the value
        if no, we supress it
    if it's above the high threshold value we consider it
    Args:
        img (np.array): the input image
        HTR (float): High Threshold Ratio, the upper limit threshold value
        LTR (float): Low Threshold Ratio, the lower limit threshold value
    Returns:
        returns the final image after performing Double Threshold Hysterisis
    '''
    GSup = np.copy(img) # create a copy of the image
    h = int(GSup.shape[0]) # get the height of the image
    w = int(GSup.shape[1]) # get the width of the image
    highThreshold = np.max(GSup) * HTR # compute the high threshold value 
    lowThreshold = highThreshold * LTR # compute the low threshold value 

    # experimenting with manually entering the threshold values
    # highThreshold = 21
    # lowThreshold = 15
    
    for i in range(1,h-1): # iterate over the rows
        for j in range(1,w-1): # iterate over the columns
            # if the pixel intensity is greater than the high threshold then we consider the pixel value
            if(GSup[i,j] > highThreshold): 
                GSup[i,j] = 1
            # else if the pixel intensity is lower than the low threshold we supress the pixel value
            elif(GSup[i,j] < lowThreshold):
                GSup[i,j] = 0
            # else we check if the neighbouring pixel values are greater than the high threshold 
            else:
                # consider the pixel value if any of the neighbouring value is greater than the high threshold
                if((GSup[i-1,j-1] > highThreshold) or 
                    (GSup[i-1,j] > highThreshold) or
                    (GSup[i-1,j+1] > highThreshold) or
                    (GSup[i,j-1] > highThreshold) or
                    (GSup[i,j+1] > highThreshold) or
                    (GSup[i+1,j-1] > highThreshold) or
                    (GSup[i+1,j] > highThreshold) or
                    (GSup[i+1,j+1] > highThreshold)):
                    GSup[i,j] = 1
        # x = np.sum(GSup == 1)
    
    # This is done to remove/clean all the weak edges which are not connected to strong edges
    GSup = (GSup == 1) * GSup 
    
    return GSup

def pad(image, kernel):
    '''
    This function adds zero-padding to the input image based on the given kernel
    Args:
        image (numpy.array): image on which the padding is to be applied.
        kernel (numpy.array): the kernel's shape is used to pad the input image
    Returns:
        an numpy.array image with zero-padding applied.
    '''
    if kernel.shape[0] == 1:
        k = kernel.shape[1]
        if k % 2 == 0:
            # if k is even
            pad_size = k
        else: # if k is odd
            pad_size = (k-1)
    
        # x direction padding
        # Add zero padding to the input image
        image_padded = np.zeros((image.shape[0] + pad_size, image.shape[1]))
        # compute center offset
        x_center = (image_padded.shape[1] - image.shape[1]) // 2
        y_center = (image_padded.shape[0] - image.shape[0]) // 2

        # copy img image into center of result image
        image_padded[y_center:y_center+image.shape[0], x_center:x_center+image.shape[1]] = image
    elif kernel.shape[1] == 1:
        k = kernel.shape[0]
        pad_size = (k-1)
    
        # y direction padding
        # Add zero padding to the input image
        image_padded = np.zeros((image.shape[0], image.shape[1] + pad_size))
        # compute center offset
        x_center = (image_padded.shape[1] - image.shape[1]) // 2
        y_center = (image_padded.shape[0] - image.shape[0]) // 2
        # copy img image into center of result image
        image_padded[y_center:y_center+image.shape[0], x_center:x_center+image.shape[1]] = image
    else:
        x_pad_size = (kernel.shape[0] - 1)
        y_pad_size = (kernel.shape[1] - 1)
        # both x and y direction padding
        # Add zero padding to the input image
        image_padded = np.zeros((image.shape[0] + x_pad_size, image.shape[1] + y_pad_size))
        # compute center offset
        x_center = (image_padded.shape[1] - image.shape[1]) // 2
        y_center = (image_padded.shape[0] - image.shape[0]) // 2
        # copy img image into center of result image
        image_padded[y_center:y_center+image.shape[0], x_center:x_center+image.shape[1]] = image
    
    return image_padded


def convolve(image, kernel):
    '''
    This function computes the convolution of the given image with the given kernel.
    Args:
        image (numpy.array): image on which the convolution is to be performed. size [image_height, image_width].
        kernel (numpy.array): the kernel which is to be applied to the image. size [kernel_height, kernel_width].
    Return: 
        a numpy array of size [image_height, image_width] (convolution output).
    '''
    # Flip the kernel
    # kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(image)
    # print(image.shape)
    image_padded = pad(image, kernel)
    # image_padded = image
    # Loop over every pixel of the image
    for x in range(image.shape[0]-kernel.shape[0]-1):
        for y in range(image.shape[1]-kernel.shape[1]-1):
            # element-wise multiplication of the kernel and the image
            # print(kernel.shape)
            # print(image_padded[x,y: y+kernel.shape[1]].shape)
            output[x, y]=(kernel * image_padded[x: x+kernel.shape[0],y: y+kernel.shape[1]]).sum()

    return output 

def canny_edge(I, sigma, HTR, LTR):
    '''
    this function performs Canny Edge Detection (steps 1-7) and returns the final image
    Args: 
        I (np.array): Image object
        sigma (float): the sigma > 0, intensity of gaussian blur
        HTR (float): High Threshold Ratio, the upper limit threshold value
        LTR (float): Low Threshold Ratio, the lower limit threshold value
    return:
        final_image (np.array): final image with Canny Edge Detection
    '''
    # step 2 and 3: create a 1D derivative array for both x and y directions, create a gaussion mask and get the derivative of the gaussian kernel 
    # This creates a Gaussian derivative mask in both x and y directions
    dx = np.array([-0.5, 0, 0.5])[np.newaxis] # transposing 1D array https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    dy = np.transpose(dx)
    
    Gx = dx * gaussmask(sigma, np.transpose(dx.shape))
    Gy = dy * gaussmask(sigma, np.transpose(dy.shape))

    # step 4: apply the gaussion derivate mask to the Image
    Ix = convolve(I, Gx)
    Iy = convolve(I, Gy)

    # step 5: Compute the magnitude of the edge response by combining the x and y components.
    I_mag = np.hypot(Ix, Iy)
    I_orient = np.degrees(np.arctan2(Iy, Ix))

    # step 6: apply Non-Max Supression
    NMS = NonMaxSup(I_mag, I_orient)
    
    # step 7: apply Double Threshold Hysterisis
    final_image = DoThreshHyst(NMS, HTR=HTR, LTR=LTR)

    return final_image
