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
    
    labelAll = nib.load(image_path[4])
    label = labelAll.get_data().astype('single')

    img_A = img_A / 4000. * 2. - 1.
    img_B = img_B / 6000. * 2. - 1.
    img_C = img_C / 10000. * 2. - 1.
    img_D = img_D / 7000. * 2. - 1.

    img_A[img_A > 1.] = 1.
    img_B[img_B > 1.] = 1.
    img_C[img_C > 1.] = 1.
    img_D[img_D > 1.] = 1.

    mask = (img_A + img_B + img_C + img_D) > -4.

    return img_A, img_B, img_C, img_D, label, mask


def load_train_data(image_path, crop_size=72, is_testing=False):
    
    img_synA = np.load(image_path[0])
    img_synB = np.load(image_path[1])
    img_synC = np.load(image_path[2])
    img_synD = np.load(image_path[3])
    label = np.load(image_path[4])
    
    imgsz = np.shape(img_synA)

    mask = (img_synA + img_synB + img_synC + img_synD) > -4.
    
    if not is_testing:
        
        combine_all = np.stack((img_synA, 
                                img_synB, 
                                img_synC, 
                                img_synD,
                                label[:,:,:,0],
                                label[:,:,:,1],
                                label[:,:,:,2],
                                mask), axis=3)
        
        h1 = int(np.ceil(random.uniform(1e-2, imgsz[0] - crop_size)))
        w1 = int(np.ceil(random.uniform(1e-2, imgsz[1] - crop_size)))
        d1 = int(np.ceil(random.uniform(1e-2, imgsz[2] - crop_size)))
        
        combine_all = combine_all[h1: h1 + crop_size, 
                                  w1: w1 + crop_size,
                                  d1: d1 + crop_size, :]
        
        # random flip x.
        if random.random() > 0.5:
            combine_all = combine_all[::-1, :, :, :]
        
        # random flip y.
        if random.random() > 0.5:
            combine_all = combine_all[:, ::-1, :, :]
            
        # random flip z.
        if random.random() > 0.5:
            combine_all = combine_all[:, :, ::-1, :]

        # random shift and random scale.
        img_synA = combine_all[:,:,:,0]
        img_synB = combine_all[:,:,:,1]
        img_synC = combine_all[:,:,:,2]
        img_synD = combine_all[:,:,:,3]
        mask = combine_all[:,:,:,7]>0.5
        
        intensity_scale = random.random() * 0.4 + 0.8
        
        intensity_shiftsynA = (random.random() * 0.4 - 0.2) * np.std(img_synA, ddof=1)
        intensity_shiftsynB = (random.random() * 0.4 - 0.2) * np.std(img_synB, ddof=1)
        intensity_shiftsynC = (random.random() * 0.4 - 0.2) * np.std(img_synC, ddof=1)
        intensity_shiftsynD = (random.random() * 0.4 - 0.2) * np.std(img_synD, ddof=1)
        
        img_synA[mask] = (img_synA[mask] + intensity_shiftsynA) * intensity_scale
        img_synB[mask] = (img_synB[mask] + intensity_shiftsynB) * intensity_scale
        img_synC[mask] = (img_synC[mask] + intensity_shiftsynC) * intensity_scale
        img_synD[mask] = (img_synD[mask] + intensity_shiftsynD) * intensity_scale
        
        combine_all[:,:,:,0] = img_synA
        combine_all[:,:,:,1] = img_synB
        combine_all[:,:,:,2] = img_synC
        combine_all[:,:,:,3] = img_synD
            
    else:
        
        combine_all = np.stack((img_synA, 
                                img_synB, 
                                img_synC, 
                                img_synD,
                                label[:,:,:,0],
                                label[:,:,:,1],
                                label[:,:,:,2],
                                mask), axis=3)
        
        h1 = int(np.ceil(random.uniform(1e-2, imgsz[0] - crop_size)))
        w1 = int(np.ceil(random.uniform(1e-2, imgsz[1] - crop_size)))
        d1 = int(np.ceil(random.uniform(1e-2, imgsz[2] - crop_size)))
        
        combine_all = combine_all[h1: h1 + crop_size, 
                                  w1: w1 + crop_size,
                                  d1: d1 + crop_size, :]
    
    return combine_all
