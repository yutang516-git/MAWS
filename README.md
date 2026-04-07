# Africa Water Body Segmentation

This is a water body segmentation task for Africa. The 'MAWS.zip' file contains a small portion of our constructed dataset, which is used for testing.

The core modules are in ./MedSAM/models/ImageEncoder/vit/adapter_fusionblock.

## Recommended directory structure
./MAWS_dataset/
├── image/          # RGB (.jpg)
│   ├── -0.009\ 12.484.jpg
│   └── ...
├── ir/             # IR (.jpg)
│   ├── -0.009\ 12.484R.jpg
│   └── ...
└── label/          # Ground Truth (.png)
    ├── -0.009\ 12.484N.png
    └── ...

## Usage

You can get the pre-trained encoder 'sam_vitb.pth' here: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints.

The fine-tuned parameter of M<sup>2</sup>FNet file is 'last-59.pth'.

Use the test.py for testing.

## References

The code is based on  [MFNet](https://github.com/sstary/SSRS.). Thanks for their great works!


