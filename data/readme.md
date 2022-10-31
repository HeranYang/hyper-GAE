## Dataset

This folder is the root data folder containing processed dataset, which needs to be downloaded and processed by users. 


In our experiments, we use the MICCAI 2019 Multimodal Brain Tumor Segmentation (BraTS 2019) and MICCAI 2018 Multimodal Brain Tumor Segmentation (BraTS 2018) datasets.
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
* For the training subset: The image intensities are further linearly scaled to [0, 1], and then the processed 3d training images are saved in .npy format to reduce the time of loading data. For a MxNxD image volume, the original segmentation label also contains MxNxD voxels, with label 4 for the enhancing tumor
(ET), label 2 for peritumoral edema (ED), label 1 for necrotic and non-enhancing tumor core (NCR/NET), and label 0 for background. We reorganize the segmentation label into a MxNxDx3 volume, where each MxNxD sub-volume respectively corresponds to the whole tumor (WT), tumor core (TC) and enhancing tumor (ET), with 1 for foreground and 0 for background.
* For the validation/test subset: The processed 3d validation and test images (without linear scaling) are saved in .nii.gz format, and the linear scaling for validation and test subjects is included in our codes within utils.py. The original segmentation labels in .nii.gz format are utilized for validation and test subset.


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
