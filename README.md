# Africa Water Body Segmentation

This is a water body segmentation task for Africa. The MAWS dataset is available at the following link:https://pan.baidu.com/s/1YkcKSJIWKEWmoNEKnXrAYg  Access code:7wes

The core modules are in ./MedSAM/models/ImageEncoder/vit/adapter_fusionblock.

## Recommended directory structure
data/MAWS_dataset/                                                                                                                                   
├── image/ # Input images (.jpg)                                                                                                                    
│ ├── sample1.jpg                                                                                                                                    
│ └── sample2.jpg                                                                                                                                    
├── ir/ # Infrared images (.jpg)                                                                                                                    
│ ├── sample1R.jpg                                                                                                                                  
│ └── sample2R.jpg                                                                                                                                   
└── label/ # Ground truth (.png)                                                                                                                     
├── sample1N.png                                                                                                                                     
└── sample2N.png

## Usage

You can get the pre-trained encoder 'sam_vitb.pth' here: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints.

Use the train.py for trainning and testing.

## References

The code is based on  [MFNet](https://github.com/sstary/SSRS). Thanks for their great works!


