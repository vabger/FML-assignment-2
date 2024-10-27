import numpy as np
from helper import *


def movePatchOverImg(image, filter_size, apply_filter_to_patch):
    grayscale_image = np.mean(image,axis=2).astype(np.uint8)

    padded_image = np.pad(grayscale_image, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)), 'constant')

    
    patch_size = filter_size
    output_image = np.zeros(grayscale_image.shape)
    
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i,j] = apply_filter_to_patch(padded_image[i:i + patch_size,j:j+patch_size])
    
    return output_image
    

def detect_horizontal_edge(image_patch):
    #ADD CODE HERE
    size = image_patch.shape[0]

    filter = np.zeros((size, size), dtype=np.float32)
    
    center = size // 2
    for i in range(size):
        if i < center: 
            filter[i, :] = 1
        elif i > center: 
            filter[i, :] = -1
    pixel_value = np.sum(filter*image_patch)
    return pixel_value

def detect_vertical_edge(image_patch):
    size = image_patch.shape[0]
    
    filter = np.zeros((size, size), dtype=np.float32)
    
    center = size // 2
    for i in range(size):
        if i < center:
            filter[:, i] = 1
        elif i > center:
            filter[:, i] = -1
    pixel_value = np.sum(filter * image_patch)
    return pixel_value

def detect_all_edges(image_patch):
    #ADD CODE HERE
    return np.sqrt(detect_horizontal_edge(image_patch)**2 + detect_vertical_edge(image_patch)**2)

def remove_noise(image_patch):
    #ADD CODE HERE
    return np.median(image_patch)

def create_gaussian_kernel(size, sigma):
    #ADD CODE HERE
    output_kernel = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            output_kernel[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-1*((i-(size-1)/2)**2 + (j-(size-1)/2)**2)/2*sigma**2)

    output_kernel /=  np.sum(output_kernel)

    return output_kernel

kernel = create_gaussian_kernel(25,1)
def gaussian_blur(image_patch):
    global kernel
    return np.sum(image_patch*kernel)

def unsharp_masking(image, scale):
    #ADD CODE HERE
    grayscale_image = np.mean(image,axis=2).astype(np.uint8)
    blurred_image = movePatchOverImg(image,25,gaussian_blur)
    high_freq = grayscale_image - blurred_image
    sharpened_image = grayscale_image + scale * high_freq
    return sharpened_image

#TASK 1  
img=load_image("cutebird.png")
filter_size=3 #You may change this to any appropriate odd number
hori_edges = movePatchOverImg(img, filter_size, detect_horizontal_edge)
save_image("hori.png",hori_edges)
filter_size=3 #You may change this to any appropriate odd number
vert_edges = movePatchOverImg(img, filter_size, detect_vertical_edge)
save_image("vert.png",vert_edges)
filter_size=3 #You may change this to any appropriate odd number
all_edges = movePatchOverImg(img, filter_size, detect_all_edges)
save_image("alledge.png",all_edges)

#TASK 2
noisyimg=load_image("noisycutebird.png")
filter_size=3#You may change this to any appropriate odd number
denoised = movePatchOverImg(noisyimg, filter_size, remove_noise)
save_image("denoised.png",denoised)


# TASK 3
scale=3 #You may use any appropriate positive number (ideally between 1 and 3)
save_image("unsharpmask.png",unsharp_masking(img,scale))
