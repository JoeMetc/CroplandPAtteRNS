# Dataset Pre-Processing

All of the dataset variants used require some level of file set up to match how we operated with them compared to their original formats.



## Tile Formatting

### PASTIS-128

This dataset needs no file alterations, but requires the added fold split files in the next step.


### PASTIS-24

This tile-size variant of PASTIS reproduces the one introduced by [TSViT](https://github.com/michaeltrs/DeepSatModels/blob/main/data/PASTIS24/data2windows.py). From each 128X128 pixel tile, 25 24X24 pixel tiles are created, discarding the last 8 pixels in each axis.

Once you have the original PASTIS dataset downloaded, create a directory `PASTIS24` at the same level as your `PASTIS` directory. Then, open `PASTIS24Writer.py` and edit lines 40 and 42 to match your directory structure. You can now run:

```
python .\source\datasets\processing\PASTIS24Writer.py
```

You will then need to copy `metadata.geojson` from the original PASTIS files to the PASTIS24 directory.


### MTLCC-24 and MTLCC-48

We have combined all pre-processing needed for MTLCC into a single script. This will sort file formats, class index mapping, and all other transforms normally applied to the raw data along with producing file structures compatible with our code. The MTLCC downloadable contains it's 24X24 pixel and 48X48 pixel tiles within a shared structure, so you will need to run this twice. We also only operate on the 2016 set in our work, but you could feasibly alter our code to work with the 2017 set instead.

To run `tfrecord2npyRemap.py` You can either alter the default dir paths at lines 313, 317, and 321, or use `--indir`, `--outdir`, and `--dates` arguments. An example command is shown here:

```
python .\source\datasets\processing\tfrecord2npyRemap.py --indir .\MTLCC\data_IJGI18\datasets\full\240\data16 --outdir .\MTLCC\data_IJGI18\datasets\fullNpy\240\data16 --dates .\MTLCC\data_IJGI18\observations.csv
```
#### You will need to run this twice, once for the `240` directory, and once for the `480` directory. Don't mix them up!



## Metadata and Fold Split File Set-Up

We use .csv files to store the official folds for both datasets. PASTIS has folds 1 through 5, and MTLCC has folds 0 through 9. MTLCC uses a shared validation set for all folds, while PASTIS uses unique sets for each fold. The .csv files have the following data structure:
```
fieldID,imagepath,labelpath
10000,PASTISFresh/DATA_S2/S2_10000.npy,PASTISFresh/ANNOTATIONS/TARGET_10000.npy
```

All official folds are provided in this format in `/foldConfs/`. For each dataset, place the included folds directory (`OfficialFolds` for PASTIS, and `folds` for MTLCC) inside the dataset root directory set in `localPaths.yaml`.


### localPaths.yaml

To specificy the location of our dataset files, we use `source/datasets/pathconfs/localPaths.yaml`. This is pre-populated with most data you will need, but still needs some alteration. For each split, you have the following fields:

`baseDir` - This is your root dataset directory for a given dataset, you will likely need to update this. All following fields are appended to this path.

`trainFile` - This is the path to the .csv holding tiles in the train split.

`valFile` - This is the path to the .csv holding tiles in the validation split.

`testFile` - This is the path to the .csv holding tiles in the test split.

`comparisonFile` - This field is currently abandoned, but was intended to place a list of tiles for two models to run on at once. Fill this field with any other valid tile .csv and it will pass over smoothly.

`metadataFile` - **PASTIS ONLY** This is the path to PASTIS' `metadata.geojson` file, which is used to derive capture dates for this dataset. This field should be deleted for MTLCC.

The split names used here are refered to by the `splits` field in the model config files.



## Example Resulting Directory Layout

If you want to minimise the required work to adapt this repo to run in your environment, these are the directory trees we ended up with for each dataset so that you can match yours to ours:

**PASTIS-128**
```
PASTIS                         # Your 'root' directory for PASTIS-128
├── PASTISFresh/               # The downloaded PASTIS dataset
|   ├── DATA_S2/               # Sentinel-2 SITS as numpy arrays
|   |   ├── S2_10000.npy       # Example SITS file
|   |   └── ...                # 
|   ├── ANNOTATIONS/           # Ground-truth segmentation masks
|   |   ├── TARGET_10000.npy   # Example labels file
|   |   └── ...                # ignore 'ParcelIDs' files
|   └── metadata.geojson       # All PASTIS tile metadata
└── OfficalFolds/  
    ├── Fold1/                 # Fold 1's split files
    |   ├── train123.csv       # Training tile IDs etc
    |   ├── val4.csv           # Validation tile IDs etc
    |   └── test5.csv          # Test tile IDs etc
    ├── Fold2/...              # Each following fold follows the above pattern
    ├── Fold3/...      
    ├── Fold4/...      
    └── Fold5/...
```

**PASTIS-24**
```
PASTIS24                       # Your 'root' directory for PASTIS-24
├── PASTISFresh/               # Named to match original
|   ├── DATA_S2/               # Named to match original
|   |   ├── S2_10000_0.npy     # Example SITS file
|   |   └── ...                # 
|   ├── ANNOTATIONS/           # Named to match original
|   |   ├── TARGET_10000_0.npy # Example labels file
|   |   └── ...                # ignore 'ParcelIDs' files
|   └── metadata.geojson       # Unaltered PASTIS tile metadata
└── OfficalFolds/              # The same as PASTIS-128, just contents of
    └── ...                    # each split file grown with -24 subtiles
```

**MTLCC-24 and MTLCC-48**
```
MTLCC                                   # Your 'root' directory for both MTLCC variants
└── data_IJGI18/                        # original MTLCC directory structure
    ├── datasets/
    │   └── fullNpy/                    # Our converted MTLCC dataset with numpy files
    │       ├── 240/
    │       │   ├── data16/
    │       │   │   ├── 1_sits.npy      # Example SITS file
    │       │   │   ├── 1_labels.npy    # Example remapped labels file
    │       │   │   ├── 1_dates.npy     # Example acquisition dates file
    │       │   │   └── ...
    │       │   └── folds/
    │       │       ├── train_fold0.csv # Training tile IDs etc
    │       │       ├── eval.csv        # Validation tile IDs etc (shared)
    │       │       ├── test_fold0.csv  # Test tile IDs etc
    │       │       └── ...             # All fold split files are in one location
    │       └── 480/
    │           ├── data16/...
    │           └── folds/...
    └── observations.csv                # backup original acquisition dates
```