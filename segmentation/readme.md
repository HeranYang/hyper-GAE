## Usage

This folder contains the code of Hyper-GAE for tumor segmentation with missing modalities. The structure of this folder is:

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
We have also uploaded our trained Hyper-GAE model for tumor segmentation with missing modalities on BraTS 2019 dataset, and one can directly use them for missing-modality tumor segmentation. Due to the restriction of github, the trained model is uploaded to the [Google Drive](https://drive.google.com/drive/folders/1LgHoGIO1h8VFuOOSe18J_38HbXeuAbVB?usp=sharing).




## Citation
If you find this code useful for your research, please cite our paper:
> @article{yang2023learning, 
> <br> title={Learning Unified Hyper-network for Multi-modal MR Image Synthesis and Tumor Segmentation with Missing Modalities}, 
> <br> author={Yang, Heran and Sun, Jian and Xu, Zongben},
> <br> journal={IEEE Transactions on Medical Imaging},
> <br> doi={10.1109/TMI.2023.3301934},
> <br> year={2023}}
