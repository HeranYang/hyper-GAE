# Hyper-GAE
This repository is the official Tensorflow implementation of the paper "Learning Unified Hyper-network for Multi-modal MR Image Synthesis and Tumor Segmentation with Missing Modalities".

Code will be published recently.

    



## Overview

The outline of this readme file is:

    Overview
    Requirements
    Dataset
    Usage
    Citation
    Reference
    
The folder structure of our implementation is:

    synthesis\       : code of Hyper-GAE for multi-modal MR image synthesis
    segmentation\    : code of Hyper-GAE for tumor segmentation with missing modalities
    data\            : root data folder (to be downloaded and preprocessed)
    


## Requirements
All experiments utilize the TensorFlow library. We recommend the following package versions:
* python == 3.6
* tensorflow-gpu == 1.10.0
* numpy == 1.19.2
* imageio == 2.9.0
* nibabel == 3.2.1



## Dataset
We use the MICCAI 2019 Multimodal Brain Tumor Segmentation (BraTS 2019) and MICCAI 2018 Multimodal Brain Tumor Segmentation (BraTS 2018) datasets in our experiments.
* [BraTS 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/): This dataset includes 335 training subjects, 125 validation subjects, and 166 test subjects, and the tumor labels of training subjects are provided. Our experiments are performed over 335 training subjects, which are randomly divided into a training set of 218 subjects, a validation set of 6 subjects, and a test set of 111 subjects.
* [BraTS 2018 dataset](https://www.med.upenn.edu/sbia/brats2018.html): This dataset contains 285 training subjects with ground-truth labels, which are split into 199, 29 and 57 subjects for training, validation and test using the same split list as in [1].


### Data Preprocessing
The data has been pre-processed by organizers, i.e., co-registered to the same anatomical template, interpolated to the same resolution and skull-stripped.
Additionally, we conduct several extra pre-processing steps:
* N4 correction
* White matter peak normalization of each modality to 1000
* Cutting out the black background area outside the brain (for both images and labels)

After preprocessing, the maximal intensities of T1w, T1ce, T2w and Flair modalities are 4000, 6000, 10000 and 7000 (arbitrary units) respectively.
Then, the training and validation/test subjects are respectively processed as follows:
* For the training subset: The image intensities are further linearly scaled to [0, 1], and then the processed 3d training images are saved in .npy format to reduce the time of loading data. For an image volume with MxNxD voxels, the original segmentation label also contains MxNxD voxels, with label 4 for the enhancing tumor
(ET), label 2 for peritumoral edema (ED), label 1 for necrotic and non-enhancing tumor core (NCR/NET), and label 0 for background. We reorganize the segmentation label into a MxNxDx3 volume, where each MxNxD sub-volume respectively corresponds to the whole tumor (WT), tumor core (TC) and enhancing tumor (ET), with 1 for foreground and 0 for background.
* For the validation/test subset: The processed 3d validation and test images are saved in .nii.gz format, and the linear scaling for validation and test subjects is included in our codes within utils.py. The segmentation labels are


### Data Folder Structure
The structure of our data folder is:

    data\    : root data folder  
        |-- BraTS-Dataset-pro\      : processed data folder
        |       |-- SliceData\      : processed 3D data in .npy format
        |       |       |-- 3DTrain\       : training data set
        |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |-- TrainB\       : t1ce  images
        |       |       |       |-- TrainC\       : t2w   images
        |       |       |       |-- TrainD\       : flair images
        |       |       |       |-- TrainL\       : segmentation labels
        |       |-- VolumeData\     : processed 3D data in .nii.gz format
        |       |       |-- Valid\         : validation data set
        |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |-- ValidB\       : t1ce  images
        |       |       |       |-- ValidC\       : t2w   images
        |       |       |       |-- ValidD\       : flair images
        |       |       |       |-- ValidL\       : segmentation labels
        |       |       |-- Test\          : test data set
        |       |       |       |-- TestA\        : t1w   images
        |       |       |       |-- TestB\        : t1ce  images
        |       |       |       |-- TestC\        : t2w   images
        |       |       |       |-- TestD\        : flair images
        |       |       |       |-- TestL\        : segmentation labels




## Usage

The structure of our code folder is:

    synthesis\         : code of Hyper-GAE for multi-modal MR image synthesis
           |-- main.py         : main function
           |-- model.py        : code of building model, and train/valid/test
           |-- module.py       : code of defining networks
           |-- ops.py          : code of defining basic components
           |-- utils.py        : code of loading train and test data
    segmentation\      : code of Hyper-GAE for tumor segmentation with missing modalities
           |-- main.py         : main function
           |-- model.py        : code of building model, and train/valid/test
           |-- module.py       : code of defining networks
           |-- ops.py          : code of defining basic components
           |-- utils.py        : code of loading train and test data


### Task I: Multi-modal MR Image Synthesis

Training phase

Valid phase

Test phase


### Task II: Brain Tumor Segmentation with Missing Modalities

Training phase

Valid phase

Test phase




## Citation
If you find this code useful for your research, please cite our paper:
> @article{yang2022learning, 
> <br> title={Learning Unified Hyper-network for Multi-modal MR Image Synthesis and Tumor Segmentation with Missing Modalities}, 
> <br> author={Yang, Heran and Sun, Jian and Xu, Zongben},
> <br> journal={Submitted to IEEE Transactions on Medical Imaging},
> <br> year={2022}}



## Reference
[1] R. Dorent, S. Joutard, M. Modat, S. Ourselin, and T. Vercauteren, “Hetero-modal variational encoder-decoder for joint modality completion and segmentation,” in MICCAI, pp. 74–82, 2019.
