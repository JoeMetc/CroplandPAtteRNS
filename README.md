# Cropland PAtteRNS


This repository contains all work to do with our paper:

**Cropland PAtteRNS: Parallel Dimensional Attention Architectures and Attention to
Dataset Dissonance for Crop Segmentation in Satellite Imagery Time Series Data**

You can access the paper here [placeholder link]

If you wish to use any of this work, or reference our paper, please condsider citing with:

```
placeholder
```

a bibtex file is also available [placeholder link to file]


## PAtteRNS Model




## Dataset Recreation

This work uses the publicly available [PASTIS](https://github.com/VSainteuf/pastis-benchmark) and [MTLCC](https://github.com/MarcCoru/MTLCC) datasets. 

The storage required for the end dataset products in this work are as follows. More storage may be required by the raw data sources or in intermediary processing steps.

| Dataset     | Tile Size Variant | File Size |
| -------     | -------           | -------   |
| PASTIS      | 128px             | 37 GB     |
| PASTIS      | 24px              | 31 GB     |
| MTLCC 2016  | 48px              | 29 GB     |
| MTLCC 2016  | 24px              | 29 GB     |


The original PASTIS Dataset tiles can be downloaded [here](https://doi.org/10.5281/zenodo.5012942).

The original MTLCC Dataset tiles can be downloaded [here](https://zenodo.org/records/5712933).

PASTIS-128, once fold split files have been configured correctly, should work as-is. PASTIS-24 and both MTLCC variants require pre-processing. Instructions for this processing, setting up fold split files, and configuring your local paths should be followed in [`source/datasets/processing/README.md`](https://github.com/JoeMetc/CroplandPAtteRNS/blob/main/source/datasets/processing/README.md).


## Running Environment

All code is written for Python 3.11.5 and developed using anaconda. To set up a conda environment to run PAtteRNS, simply use the included `environment.yml` file:

```
conda env create -f .\environment.yml

conda activate PAtteRNS
```

## Model Use

Once the instructions above have been followed to set up at least one dataset along with the runtime environment, the PAtteRNS model is ready to use. The following operations of the model have been provisioned:

#### Running Pre-Trained Models


#### Training PAtteRNS Yourself


#### Training Other Models Yourself

This repo also contains everything necessary to train [TSViT](), [UTAE](), and [UNet3D]() on any of our supported dataset variants. Configuration files are already set up for these in `source/configs`,