#####
# Author: Joseph Metcalfe
# Purpose: Writing fold population tile lists to .csv files
#####

import pandas as pd
import geopandas as gpd
import os
import re

WORKINGDIR = "/home/s.2275333/Datasets/PASTIS" # "/home/s.2275333/Datasets/PASTIS", "D:/Data/PASTIS24", "D:/Data/PASTIS"
SAVEDIR = "OfficialFolds"
USEDFOLDS = [1,2,3,4,5]

metadataPath = "PASTISFresh/metadata.geojson"

tilesMetadataTable = gpd.read_file(os.path.join(WORKINGDIR,metadataPath))
tilesMetadataTable["id"] = tilesMetadataTable["id"].astype(str)


# get a df together of the actual info to go in the fold csv files
allRawFiles = os.listdir(os.path.join(WORKINGDIR, "PASTISFresh/DATA_S2"))

id_re = re.compile(r"S2_(\d+)")
allTileIDs = [id_re.search(f).group(1) for f in allRawFiles if id_re.search(f)]
allDataPaths = list(map(lambda id: f"PASTISFresh/DATA_S2/S2_{id}.npy", allTileIDs))
allLabelPaths = list(map(lambda id: f"PASTISFresh/ANNOTATIONS/TARGET_{id}.npy", allTileIDs))

allSetsData = pd.DataFrame(columns=["fieldID","imagepath","labelpath"])

allSetsData["fieldID"] = allTileIDs
allSetsData["imagepath"] = allDataPaths
allSetsData["labelpath"] = allLabelPaths
# eliminate any rows where one of the files doesnt exist
allSetsData = allSetsData[[os.path.isfile(os.path.join(WORKINGDIR, i)) for i in allSetsData['labelpath']]]


# structure for gettign right tiles in right place per fold
foldDFList = [tilesMetadataTable[tilesMetadataTable['Fold'] == fold]
              for fold in sorted(tilesMetadataTable['Fold'].unique())]
tilesFold1, tilesFold2, tilesFold3, tilesFold4, tilesFold5 = foldDFList

fold1IDs = list(tilesFold1["id"])
fold2IDs = list(tilesFold2["id"])
fold3IDs = list(tilesFold3["id"])
fold4IDs = list(tilesFold4["id"])
fold5IDs = list(tilesFold5["id"])

foldIDsList = [fold1IDs, fold2IDs, fold3IDs, fold4IDs, fold5IDs]

for rootFold in USEDFOLDS:
    index = rootFold-1
    trainList = [USEDFOLDS[index], USEDFOLDS[(index+1)%5], USEDFOLDS[(index+2)%5]]
    valFold = USEDFOLDS[(index+3)%5]
    testFold = USEDFOLDS[(index+4)%5]

    trainIDs = [foldIDsList[trainList[0]-1],foldIDsList[trainList[1]-1],foldIDsList[trainList[2]-1]]
    trainIDs = [tileID for foldIDs in trainIDs for tileID in foldIDs]
    valIDs = foldIDsList[valFold-1]
    testIDs = foldIDsList[testFold-1]

    trainSet = allSetsData[allSetsData['fieldID'].isin(trainIDs)]
    valSet = allSetsData[allSetsData['fieldID'].isin(valIDs)]
    testSet = allSetsData[allSetsData['fieldID'].isin(testIDs)]

    # make the fold sub-directory
    subDir = os.path.join(os.path.join(WORKINGDIR,SAVEDIR),f"Fold{rootFold}")
    os.mkdir(subDir)

    # set out file names
    trainFile = os.path.join(subDir,f"train{trainList[0]}{trainList[1]}{trainList[2]}.csv")
    valFile = os.path.join(subDir,f"val{valFold}.csv")
    testFile = os.path.join(subDir,f"test{testFold}.csv")

    # write files
    trainSet.to_csv(trainFile, index=False)
    valSet.to_csv(valFile, index=False)
    testSet.to_csv(testFile, index=False)






