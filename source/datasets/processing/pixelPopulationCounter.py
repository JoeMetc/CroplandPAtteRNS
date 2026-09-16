#####
# Author: Joseph Metcalfe
# Creation Date: 06/05/2025
# Purpose: Checking populations of different classes on a total pixel area basis
#####

import pandas as pd
import numpy as np
from os.path import join
from yaml import safe_load

PASTIS_DATASET = "PASTIS24Fold1"
MTLCC_DATASET = "MTLCC24Fold0"

################ PASTIS #################

with open("source/configs/PAtteRNS-PASTIS-24.yaml", 'r') as configFile:
    configPastis = safe_load(configFile)

with open(configPastis["LOCAL"]["datasetPathsFile"], 'r') as datasetsConfigFile:
    datasetsConfigPastis = safe_load(datasetsConfigFile)

baseDirPastis = datasetsConfigPastis[PASTIS_DATASET]["baseDir"]
trainSetPastis = pd.read_csv(join(baseDirPastis,datasetsConfigPastis[PASTIS_DATASET]["trainFile"]))
valSetPastis = pd.read_csv(join(baseDirPastis,datasetsConfigPastis[PASTIS_DATASET]["valFile"]))
testSetPastis = pd.read_csv(join(baseDirPastis,datasetsConfigPastis[PASTIS_DATASET]["testFile"]))

labelPathsPastis = []
for _, field in trainSetPastis.iterrows():
    labelPathsPastis.append(field[2])
for _, field in valSetPastis.iterrows():
    labelPathsPastis.append(field[2])
for _, field in testSetPastis.iterrows():
    labelPathsPastis.append(field[2])

pastisClasses = ["Background", "Meadow", "Soft Winter Wheat", "Corn", "Winter Barley", "Winter Rapeseed", "Spring Barley", "Sunflower", "Grapevine", "Beet", "Winter Triticale", "Winter Durum Wheat", "Fruits, Vegtables, & Flowers", "Potatoes", "Leguminous Fodder", "Soybeans", "Orchard", "Mixed Cereal", "Sorghum", "Void Parcel"]
populationDictPastis = {value: 0 for _, value in enumerate(pastisClasses)}
proportionDictPastis = {value: 0 for _, value in enumerate(pastisClasses)}

for labelFile in labelPathsPastis:
    labelsImg = np.load(join(baseDirPastis,labelFile))[0]
    classes, counts = np.unique(labelsImg, return_counts=True)
    # for every unique class in this label image, update the total number of pixels in the whole dataset totals
    for updatingClass in range(0, len(classes)):
        populationDictPastis[pastisClasses[classes[updatingClass]]] += counts[updatingClass]

# print(f"Raw Class Pixel Populations: {populationDict}")

totalPixels = len(labelPathsPastis) * 24 * 24

# create dict with proportions out of dict with totals
for pastisClass in pastisClasses:
    proportionDictPastis[pastisClass] = populationDictPastis[pastisClass] / totalPixels

# print(f"Class Proportions: {proportionDict}")

# put it all in an output for reading by other files when needed
classData = pd.DataFrame(columns=["Class","Pixels","Proportion","InversedProportionWeight"])

for pastisClass in pastisClasses:
    classData.loc[len(classData)] = [pastisClass, populationDictPastis[pastisClass], proportionDictPastis[pastisClass], 1/proportionDictPastis[pastisClass]]

print("PASTIS:\n\n\n")
print(classData)

classData.to_csv(join(baseDirPastis, "PASTIS24FullClassStatsPAtteRNS.csv"), index=False)

################ MTLCC #################

with open("source/configs/PAtteRNS-MTLCC-24.yaml", 'r') as configFile:
    configMtlcc = safe_load(configFile)

with open(configMtlcc["LOCAL"]["datasetPathsFile"], 'r') as datasetsConfigFile:
    datasetsConfigMtlcc = safe_load(datasetsConfigFile)

baseDirMtlcc = datasetsConfigMtlcc[MTLCC_DATASET]["baseDir"]
trainSetMtlcc = pd.read_csv(join(baseDirMtlcc,datasetsConfigMtlcc[MTLCC_DATASET]["trainFile"]))
valSetMtlcc = pd.read_csv(join(baseDirMtlcc,datasetsConfigMtlcc[MTLCC_DATASET]["valFile"]))
testSetMtlcc = pd.read_csv(join(baseDirMtlcc,datasetsConfigMtlcc[MTLCC_DATASET]["testFile"]))

labelPathsMtlcc = []
for _, field in trainSetMtlcc.iterrows():
    labelPathsMtlcc.append(field[2])
for _, field in valSetMtlcc.iterrows():
    labelPathsMtlcc.append(field[2])
for _, field in testSetMtlcc.iterrows():
    labelPathsMtlcc.append(field[2])

print(len(labelPathsMtlcc))

MtlccClasses = ["Unknown", "Sugar Beet", "Summer Oat", "Meadow", "Rape", "Hop", "Winter Spelt", "Winter Triticale", "Beans", "Peas", "Potatoe", "Soybeans", "Asparagus", "Winter Wheat", "Winter Barley", "Winter Rye", "Summer Barley", "Maize"]
populationDictMtlcc = {value: 0 for _, value in enumerate(MtlccClasses)}
proportionDictMtlcc = {value: 0 for _, value in enumerate(MtlccClasses)}

for labelFile in labelPathsMtlcc:
    labelsImg = np.load(join(baseDirMtlcc,labelFile))
    classes, counts = np.unique(labelsImg, return_counts=True)
    # for every unique class in this label image, update the total number of pixels in the whole dataset totals
    for updatingClass in range(0, len(classes)):
        populationDictMtlcc[MtlccClasses[classes[updatingClass]]] += counts[updatingClass]

# print(f"Raw Class Pixel Populations: {populationDict}")

totalPixels = len(labelPathsMtlcc) * 24 * 24

# create dict with proportions out of dict with totals
for MtlccClass in MtlccClasses:
    proportionDictMtlcc[MtlccClass] = populationDictMtlcc[MtlccClass] / totalPixels

# print(f"Class Proportions: {proportionDict}")

# put it all in an output for reading by other files when needed
classData = pd.DataFrame(columns=["Class","Pixels","Proportion","InversedProportionWeight"])

for MtlccClass in MtlccClasses:
    classData.loc[len(classData)] = [MtlccClass, populationDictMtlcc[MtlccClass], proportionDictMtlcc[MtlccClass], 1/proportionDictMtlcc[MtlccClass]]

print("MTLCC:\n\n\n")
print(classData)

classData.to_csv(join(baseDirMtlcc, "MTLCC24FullClassStatsPAtteRNS.csv"), index=False)