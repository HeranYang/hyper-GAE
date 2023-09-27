from __future__ import division
import numpy as np
import nibabel as nib
import random



def load_test_data(image_path):

    imgAll_A = nib.load(image_path[0])
    img_A = imgAll_A.get_data().astype('single')
    
    imgAll_B = nib.load(image_path[1])
    img_B = imgAll_B.get_data().astype('single')
    
    imgAll_C = nib.load(image_path[2])
    img_C = imgAll_C.get_data().astype('single')
    
    imgAll_D = nib.load(image_path[3])
    img_D = imgAll_D.get_data().astype('single')

    img_A = img_A / 4000. * 2. - 1.
    img_B = img_B / 6000. * 2. - 1.
    img_C = img_C / 10000. * 2. - 1.
    img_D = img_D / 7000. * 2. - 1.

    img_A[img_A > 1.] = 1.
    img_B[img_B > 1.] = 1.
    img_C[img_C > 1.] = 1.
    img_D[img_D > 1.] = 1.

    return img_A, img_B, img_C, img_D



def load_train_data(image_path, crop_size=72, is_testing=False):
    
    img_synA = np.load(image_path[0])
    img_synB = np.load(image_path[1])
    img_synC = np.load(image_path[2])
    img_synD = np.load(image_path[3])
    
    imgsz = np.shape(img_synA)

    mask = (img_synA + img_synB + img_synC + img_synD) > -4.
    
    if not is_testing:
        
        combine_all = np.stack((img_synA, 
                                img_synB, 
                                img_synC, 
                                img_synD,
                                mask), axis=3)
        
        h1 = int(np.ceil(random.uniform(1e-2, imgsz[0] - crop_size)))
        w1 = int(np.ceil(random.uniform(1e-2, imgsz[1] - crop_size)))
        d1 = int(np.ceil(random.uniform(1e-2, imgsz[2] - crop_size)))
        
        combine_all = combine_all[h1: h1 + crop_size, 
                                  w1: w1 + crop_size,
                                  d1: d1 + crop_size, :]

        img_synA = combine_all[:,:,:,0]
        img_synB = combine_all[:,:,:,1]
        img_synC = combine_all[:,:,:,2]
        img_synD = combine_all[:,:,:,3]
        
        combine_all = np.stack((img_synA, 
                                img_synB, 
                                img_synC, 
                                img_synD), axis=3)
            
    else:
        
        combine_all = np.stack((img_synA, 
                                img_synB, 
                                img_synC, 
                                img_synD), axis=3)
        
        h1 = int(np.ceil(random.uniform(1e-2, imgsz[0] - crop_size)))
        w1 = int(np.ceil(random.uniform(1e-2, imgsz[1] - crop_size)))
        d1 = int(np.ceil(random.uniform(1e-2, imgsz[2] - crop_size)))
        
        combine_all = combine_all[h1: h1 + crop_size, 
                                  w1: w1 + crop_size,
                                  d1: d1 + crop_size, :]
    
    return combine_all
