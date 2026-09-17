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


### PAtteRNS Model




### Dataset Recreation

This work uses the publicly available [PASTIS](https://github.com/VSainteuf/pastis-benchmark) and [MTLCC](https://github.com/MarcCoru/MTLCC) datasets. 

The storage required for the end dataset products in this work are as follows. More storage may be required by the raw data sources or in intermediary processing steps.

| Dataset     | Tile Size Variant | File Size |
| -------     | -------           | -------   |
| PASTIS      | 128px             | 37 GB     |
| PASTIS      | 24px              | 31 GB     |
| MTLCC 2016  | 48px              | 29 GB     |
| MTLCC 2016  | 24px              | 29 GB     |

#### PASTIS

The original PASTIS Dataset tiles can be downloaded [here](https://doi.org/10.5281/zenodo.5012942)


#### PASTIS-24


#### MTLCC

The original MTLCC Dataset tiles can be downloaded [here](https://zenodo.org/records/5712933)


### Running Environment

All code is written for Python 3.11.5 and developed using anaconda. To set up a conda environment to run PAtteRNS, simply use the included `environment.yml` file:

```
conda env create -f .\environment.yml

conda activate PAtteRNS
```