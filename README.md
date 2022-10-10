# hyper-GAE
This repository is the official Tensorflow implementation of the paper "Learning Unified Hyper-network for Multi-modal MR Image Synthesis and Tumor Segmentation with Missing Modalities".

Code will be published recently.



## Overview



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
* Cutting out the black background area outside the brain

After preprocessing, the maximal intensities of T1w, T1ce, T2w and Flair modalities are 4000, 6000, 10000 and 7000 (arbitrary units) respectively.

### Folder Structure
The structure of our data folder is:

    data\    : Root data folder  
        |-- BraTS-Dataset-pro\      : processed data folder
        |       |-- SliceData\      : processed 3D data in .npy format
        |       |       |-- 3DTrain\       : training data set
        |       |       |       |-- TrainA\       : T1w   images
        |       |       |       |-- TrainB\       : T1ce  images
        |       |       |       |-- TrainC\       : T2w   images
        |       |       |       |-- TrainD\       : flair images
        |       |       |       |-- TrainL\       : segmentation labels
        |       |-- VolumeData\     : processed 3D data in .nii.gz format
        |       |       |-- Valid\         : validation data set
        |       |       |       |-- ValidA\       : T1w   images
        |       |       |       |-- ValidB\       : T1ce  images
        |       |       |       |-- ValidC\       : T2w   images
        |       |       |       |-- ValidD\       : flair images
        |       |       |       |-- ValidL\       : segmentation labels
        |       |       |-- Test\          : test data set
        |       |       |       |-- TestA\        : T1w   images
        |       |       |       |-- TestB\        : T1ce  images
        |       |       |       |-- TestC\        : T2w   images
        |       |       |       |-- TestD\        : flair images
        |       |       |       |-- TestL\        : segmentation labels




## TASK 1: Multi-modal MR Image Synthesis


### Training Phase


### Valid Phase


### Test Phase



## TASK 2: Brain Tumor Segmentation with Missing Modalities


### Training Phase


### Valid Phase


### Test Phase




## Citation
If you use this code for your research, please cite our paper:
> @article{yang2022learning, 
> <br> title={Learning Unified Hyper-network for Multi-modal MR Image Synthesis and Tumor Segmentation with Missing Modalities}, 
> <br> author={Yang, Heran and Sun, Jian and Xu, Zongben},
> <br> booktitle={Submitted to IEEE Transactions on Medical Imaging},
> <br>}



## Reference
[1] R. Dorent, S. Joutard, M. Modat, S. Ourselin, and T. Vercauteren, “Hetero-modal variational encoder-decoder for joint modality completion and segmentation,” in MICCAI, pp. 74–82, 2019.
