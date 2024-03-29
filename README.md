# Hyper-GAE
This repository is the official Tensorflow implementation of the paper "Learning Unified Hyper-network for Multi-modal MR Image Synthesis and Tumor Segmentation with Missing Modalities" in IEEE TMI 2023.
    



## Overview

<div align=center>
<img src="https://github.com/HeranYang/hyper-GAE/blob/main/images/Framework.png" width="800px"/>
</div>


Accurate segmentation of brain tumors is of critical importance in clinical assessment and treatment planning, which requires multiple MR modalities providing complementary information. However, due to practical limits, one or more modalities may be missing in real scenarios. To tackle this problem, existing methods need to train multiple networks or a unified but fixed network for various possible missing modality cases, which leads to high computational burdens or sub-optimal performance. In this paper, we propose a unified and adaptive multimodal MR image synthesis method, and further apply it to tumor segmentation with missing modalities. Based on the decomposition of multi-modal MR images into common and modality-specific features, we design a shared hyper-encoder for embedding each available modality into the feature space, a graph-attention-based fusion block to aggregate the features of available modalities to the fused features, and a shared hyper-decoder for image reconstruction. We also propose an adversarial common feature constraint to enforce the fused features to be in a common space. As for missing modality segmentation, we first conduct the feature-level and image-level completion using our synthesis method and then segment the tumors based on the completed MR images together with the extracted
common features. Moreover, we design a hypernet-based modulation module to adaptively utilize the real and synthetic modalities. Experimental results suggest that our method can not only synthesize reasonable multi-modal MR images, but also achieve state-of-the-art performance on brain tumor segmentation with missing modalities.


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
* nvidia-tensorflow == 1.15.5
* numpy == 1.19.5
* imageio == 2.9.0
* nibabel == 4.0.2



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

* For the *training* subset
    * *Image*: The image intensities are further linearly scaled to [-1, 1], and then the processed 3d training images are saved in .npy format to reduce the time of loading data. 
    * *Label*: For a MxNxD image volume, the original segmentation label also contains MxNxD voxels, with label 4 for the enhancing tumor
(ET), label 2 for peritumoral edema (ED), label 1 for necrotic and non-enhancing tumor core (NCR/NET), and label 0 for background. We reorganize the segmentation label into a MxNxDx3 volume, where each MxNxD sub-volume respectively corresponds to the whole tumor (WT), tumor core (TC) and enhancing tumor (ET), with 1 for foreground and 0 for background.

* For the *validation & test* subset
   * *Image*: The processed 3d validation and test images (without linear scaling) are saved in .nii.gz format, and the linear scaling for validation and test subjects is included in our codes within utils.py. 
   * *Label*: The original segmentation labels in .nii.gz format are utilized for validation and test subset.


### Data Folder Structure
The structure of our data folder is:

    data\    : root data folder  
        |-- BraTS-Dataset-pro\      : processed data folder for BraTS 2019 dataset
        |       |-- SliceData\         : processed 3D data in .npy format
        |       |       |-- 3DTrain\       : training data set
        |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |       |-- BraTS19-id{:0>3d}.npy       : image name format
        |       |       |       |-- TrainB\       : t1ce  images
        |       |       |       |-- TrainC\       : t2w   images
        |       |       |       |-- TrainD\       : flair images
        |       |       |       |-- TrainL\       : segmentation labels
        |       |-- VolumeData\        : processed 3D data in .nii.gz format
        |       |       |-- Valid\         : validation data set
        |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |       |-- BraTS19-id{:0>3d}.nii.gz    : image name format
        |       |       |       |-- ValidB\       : t1ce  images
        |       |       |       |-- ValidC\       : t2w   images
        |       |       |       |-- ValidD\       : flair images
        |       |       |       |-- ValidL\       : segmentation labels
        |       |       |-- Test\          : test data set
        |       |       |       |-- TestA\        : t1w   images
        |       |       |       |       |-- BraTS19-id{:0>3d}.nii.gz    : image name format
        |       |       |       |-- TestB\        : t1ce  images
        |       |       |       |-- TestC\        : t2w   images
        |       |       |       |-- TestD\        : flair images
        |       |       |       |-- TestL\        : segmentation labels
        |-- BraTS-2018\             : processed data folder for BraTS 2018 dataset
        |       |-- Fold1\             : fold-1 data of three-fold cross-validation
        |       |       |-- npyData\       : processed 3D data in .npy format
        |       |       |       |-- Train\        : training data set
        |       |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.npy       : image name format
        |       |       |       |       |-- TrainB\       : t1ce  images
        |       |       |       |       |-- TrainC\       : t2w   images
        |       |       |       |       |-- TrainD\       : flair images
        |       |       |       |       |-- TrainL\       : segmentation labels
        |       |       |-- niiData\       : processed 3D data in .nii.gz format
        |       |       |       |-- Valid\         : validation data set
        |       |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.nii.gz    : image name format
        |       |       |       |       |-- ValidB\       : t1ce  images
        |       |       |       |       |-- ValidC\       : t2w   images
        |       |       |       |       |-- ValidD\       : flair images
        |       |       |       |       |-- ValidL\       : segmentation labels
        |       |       |       |-- Test\          : test data set
        |       |       |       |       |-- TestA\        : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.nii.gz    : image name format
        |       |       |       |       |-- TestB\        : t1ce  images
        |       |       |       |       |-- TestC\        : t2w   images
        |       |       |       |       |-- TestD\        : flair images
        |       |       |       |       |-- TestL\        : segmentation labels
        |       |-- Fold2\             : fold-2 data of three-fold cross-validation
        |       |       |-- npyData\       : processed 3D data in .npy format
        |       |       |       |-- Train\        : training data set
        |       |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.npy       : image name format
        |       |       |       |       |-- TrainB\       : t1ce  images
        |       |       |       |       |-- TrainC\       : t2w   images
        |       |       |       |       |-- TrainD\       : flair images
        |       |       |       |       |-- TrainL\       : segmentation labels
        |       |       |-- niiData\       : processed 3D data in .nii.gz format
        |       |       |       |-- Valid\         : validation data set
        |       |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.nii.gz    : image name format
        |       |       |       |       |-- ValidB\       : t1ce  images
        |       |       |       |       |-- ValidC\       : t2w   images
        |       |       |       |       |-- ValidD\       : flair images
        |       |       |       |       |-- ValidL\       : segmentation labels
        |       |       |       |-- Test\          : test data set
        |       |       |       |       |-- TestA\        : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.nii.gz    : image name format
        |       |       |       |       |-- TestB\        : t1ce  images
        |       |       |       |       |-- TestC\        : t2w   images
        |       |       |       |       |-- TestD\        : flair images
        |       |       |       |       |-- TestL\        : segmentation labels
        |       |-- Fold3\             : fold-3 data of three-fold cross-validation
        |       |       |-- npyData\       : processed 3D data in .npy format
        |       |       |       |-- Train\        : training data set
        |       |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.npy       : image name format
        |       |       |       |       |-- TrainB\       : t1ce  images
        |       |       |       |       |-- TrainC\       : t2w   images
        |       |       |       |       |-- TrainD\       : flair images
        |       |       |       |       |-- TrainL\       : segmentation labels
        |       |       |-- niiData\       : processed 3D data in .nii.gz format
        |       |       |       |-- Valid\         : validation data set
        |       |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.nii.gz    : image name format
        |       |       |       |       |-- ValidB\       : t1ce  images
        |       |       |       |       |-- ValidC\       : t2w   images
        |       |       |       |       |-- ValidD\       : flair images
        |       |       |       |       |-- ValidL\       : segmentation labels
        |       |       |       |-- Test\          : test data set
        |       |       |       |       |-- TestA\        : t1w   images
        |       |       |       |       |       |-- BraTS18-id{:0>3d}.nii.gz    : image name format
        |       |       |       |       |-- TestB\        : t1ce  images
        |       |       |       |       |-- TestC\        : t2w   images
        |       |       |       |       |-- TestD\        : flair images
        |       |       |       |       |-- TestL\        : segmentation labels




## Usage

We provide the codes of our Hyper-GAE respectively for multi-modal MR image synthesis and tumor segmentation with missing modalities.
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


### Training

Our code can be trained using the following commond:

    python main.py --batch_size=2 --phase=train

If you want to continue train the model, you could uncomment the continue_training codes and comment the warmup strategy codes in train function within model.py, and then run the commond above.


### Validation

Before starting the validation process, you may need to modify the information about valid set and epoch in valid function within model.py.
Then, the validation process can be conducted using the following commond:

    python main.py --batch_size=1 --phase=valid
    
After generating the validation results, you could select the optimal epoch_id based on the performance on validation set.


### Test

Before starting the test process, you need to set the epoch as the selected optimal epoch_id in test function within model.py.
Then, you can generate the test results using the following commond:

    python main.py --batch_size=1 --phase=test

Note that our codes defaultly utilize the 8-direction flips during inference, and you could comment the codes of flips 2-8 if you do not want to use this strategy.


### About Trained Model
We have also uploaded our trained Hyper-GAE models for multi-modal MR image synthesis and tumor segmentation with missing modalities on BraTS 2019 dataset, and one can directly use them for missing-modality MR image synthesis and segmentation. Due to the restriction of github, the trained models are uploaded to the [Google Drive](https://drive.google.com/drive/folders/1L0WFTrmLwCDLOv0l5Ml2pjyFcGoe-PKg?usp=sharing).



## Citation
If you find this code useful for your research, please cite our paper:
> @article{yang2023learning, 
> <br> title={Learning Unified Hyper-network for Multi-modal MR Image Synthesis and Tumor Segmentation with Missing Modalities}, 
> <br> author={Yang, Heran and Sun, Jian and Xu, Zongben},
> <br> journal={IEEE Transactions on Medical Imaging},
> <br> doi={10.1109/TMI.2023.3301934},
> <br> year={2023}}



## Reference
[1] R. Dorent, S. Joutard, M. Modat, S. Ourselin, and T. Vercauteren, “Hetero-modal variational encoder-decoder for joint modality completion and segmentation,” in MICCAI, pp. 74–82, 2019.
