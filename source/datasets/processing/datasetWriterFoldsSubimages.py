#####
# Author: Joseph Metcalfe
# Purpose: Writing fold population tile lists to .csv files (specifically PASTIS 24)
#####

import pandas as pd
import geopandas as gpd
import os
import re

WORKINGDIR = "D:/Data/PASTIS24"
SAVEDIR = "OfficialFolds"
USEDFOLDS = [1, 2, 3, 4, 5]

metadataPath = "PASTISFresh/metadata.geojson"
tilesMetadataTable = gpd.read_file(os.path.join(WORKINGDIR, metadataPath))

# list all sub-tile files
data_dir = os.path.join(WORKINGDIR, "PASTISFresh", "DATA_S2")
label_dir = os.path.join(WORKINGDIR, "PASTISFresh", "ANNOTATIONS")
allDataFiles = sorted(os.listdir(data_dir))
allLabelFiles = sorted(os.listdir(label_dir))

# helper to extract patch id and tile index from filenames like "S2_10000_0.npy"
pat_data = re.compile(r"^S2_(\d{4,})_(\d+)\.npy$")
pat_label = re.compile(r"^TARGET_(\d{4,})_(\d+)\.npy$")

# Build a mapping: patch_id -> list of data file relative paths
patch_to_datafiles = {}
for fname in allDataFiles:
    m = pat_data.match(fname)
    if not m:
        continue
    pid, tid = m.group(1), int(m.group(2))
    patch_to_datafiles.setdefault(pid, []).append((tid, os.path.join("PASTISFresh", "DATA_S2", fname)))

# Build mapping for labels (to check existence)
patch_to_labelfiles = {}
for fname in allLabelFiles:
    m = pat_label.match(fname)
    if not m:
        continue
    pid, tid = m.group(1), int(m.group(2))
    patch_to_labelfiles.setdefault(pid, set()).add(tid)

# Prepare allSetsData rows
rows = []
for pid, data_list in patch_to_datafiles.items():
    # sort by tile index so ordering is deterministic
    data_list.sort(key=lambda x: x[0])  # list of (tile_idx, relpath)
    for tile_idx, rel_img_path in data_list:
        # check corresponding label exists
        if tile_idx in patch_to_labelfiles.get(pid, set()):
            rel_label_path = os.path.join("PASTISFresh", "ANNOTATIONS", f"TARGET_{pid}_{tile_idx}.npy")
            subtileID = f"{pid}_{tile_idx}"
            rows.append({"fieldID": subtileID, "imagepath": rel_img_path, "labelpath": rel_label_path, "parentID": pid})
        else:
            # skip if label missing
            # (could also choose to warn)
            continue

allSetsData = pd.DataFrame(rows)

# prepare fold IDs list in the same order as USEDFOLDS
foldDFList = [tilesMetadataTable[tilesMetadataTable['Fold'] == fold] for fold in USEDFOLDS]
foldIDsList = [list(df['id'].astype(str)) for df in foldDFList]  # ensure string IDs

# write CSVs per fold
for rootFold in USEDFOLDS:
    index = rootFold - 1
    trainList = [USEDFOLDS[index], USEDFOLDS[(index + 1) % 5], USEDFOLDS[(index + 2) % 5]]
    valFold = USEDFOLDS[(index + 3) % 5]
    testFold = USEDFOLDS[(index + 4) % 5]

    # flatten train IDs
    trainIDs = []
    for t in trainList:
        trainIDs.extend(foldIDsList[t - 1])

    valIDs = foldIDsList[valFold - 1]
    testIDs = foldIDsList[testFold - 1]

    trainSet = allSetsData[allSetsData['parentID'].isin(trainIDs)]
    valSet = allSetsData[allSetsData['parentID'].isin(valIDs)]
    testSet = allSetsData[allSetsData['parentID'].isin(testIDs)]

    subDir = os.path.join(WORKINGDIR, SAVEDIR, f"Fold{rootFold}")
    os.makedirs(subDir, exist_ok=True)

    trainFile = os.path.join(subDir, f"train{trainList[0]}{trainList[1]}{trainList[2]}.csv")
    valFile = os.path.join(subDir, f"val{valFold}.csv")
    testFile = os.path.join(subDir, f"test{testFold}.csv")

    trainSet.to_csv(trainFile, index=False)
    valSet.to_csv(valFile, index=False)
    testSet.to_csv(testFile, index=False)

print("Fold CSVs written successfully.")