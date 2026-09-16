#####
# Author: Joseph Metcalfe
# Creation Date: 24/02/2025
# Purpose: Initialise dataset module with generic get dataloader functionality to call specific dataset loaders
#####

import pandas as pd
from os.path import join
from datasets.PASTIS128Loader import PASTIS128Dataloader
from datasets.PASTIS24Loader import PASTIS24Dataloader
from datasets.MTLCC48Loader import MTLCC48Dataloader
from datasets.MTLCC24Loader import MTLCC24Dataloader
from utils.configIO import yamlRead

def SITSDataloader(config, seed):

    localPaths = yamlRead(config["LOCAL"]["datasetPathsFile"])

    maskBackground = config["MODEL"]["maskBackground"]

    dataset = config["DATA"]["dataset"]
    folds = config["DATA"]["folds"]
    splits = config["DATA"]["splits"]
    batchSize = config["DATA"]["batchSize"]
    workers = config["DATA"]["numWorkers"]

    foldLoaders = []

    for fold in range(0,folds):
        splitName = splits[fold]
        splitDataloaders = {}
        
        print(f"Loading Fold {fold}")

        if dataset == "PASTIS":

            rootDir = localPaths[splitName]["baseDir"]
            trainPathsFile = localPaths[splitName]["trainFile"]
            valPathsFile = localPaths[splitName]["valFile"]
            testPathsFile = localPaths[splitName]["testFile"]
            comparisonPathsFile = localPaths[splitName]["comparisonFile"]
            metadataPath = localPaths[splitName]["metadataFile"]
            # Train
            splitDataloaders['train'] = PASTIS128Dataloader(config, rootDir, trainPathsFile, metadataPath, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=True, seed = seed) # pass in paths to data
            # Validate
            splitDataloaders['validate'] = PASTIS128Dataloader(config, rootDir, valPathsFile, metadataPath, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle = False, seed = seed) # pass in paths to data
            # Test
            splitDataloaders['test'] = PASTIS128Dataloader(config, rootDir, testPathsFile, metadataPath, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data
            # Comparison
            splitDataloaders['comparison'] = PASTIS128Dataloader(config, rootDir, comparisonPathsFile, metadataPath, maskBackground, batchSize=1, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data

        elif dataset == "PASTIS24":

            rootDir = localPaths[splitName]["baseDir"]
            trainPathsFile = localPaths[splitName]["trainFile"]
            valPathsFile = localPaths[splitName]["valFile"]
            testPathsFile = localPaths[splitName]["testFile"]
            comparisonPathsFile = localPaths[splitName]["comparisonFile"]
            metadataPath = localPaths[splitName]["metadataFile"]
            # Train
            splitDataloaders['train'] = PASTIS24Dataloader(config, rootDir, trainPathsFile, metadataPath, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=True, seed = seed) # pass in paths to data
            # Validate
            splitDataloaders['validate'] = PASTIS24Dataloader(config, rootDir, valPathsFile, metadataPath, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle = False, seed = seed) # pass in paths to data
            # Test
            splitDataloaders['test'] = PASTIS24Dataloader(config, rootDir, testPathsFile, metadataPath, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data
            # Comparison
            splitDataloaders['comparison'] = PASTIS24Dataloader(config, rootDir, comparisonPathsFile, metadataPath, maskBackground, batchSize=1, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data

        elif dataset == "MTLCC48":

            rootDir = localPaths[splitName]["baseDir"]
            trainPathsFile = localPaths[splitName]["trainFile"]
            valPathsFile = localPaths[splitName]["valFile"]
            testPathsFile = localPaths[splitName]["testFile"]
            comparisonPathsFile = localPaths[splitName]["comparisonFile"]
            # Train
            splitDataloaders['train'] = MTLCC48Dataloader(config, rootDir, trainPathsFile, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=True, seed = seed) # pass in paths to data
            # Validate
            splitDataloaders['validate'] = MTLCC48Dataloader(config, rootDir, valPathsFile, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle = False, seed = seed) # pass in paths to data
            # Test
            splitDataloaders['test'] = MTLCC48Dataloader(config, rootDir, testPathsFile, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data
            # Comparison
            splitDataloaders['comparison'] = MTLCC48Dataloader(config, rootDir, comparisonPathsFile, maskBackground, batchSize=1, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data

        elif dataset == "MTLCC24":

            rootDir = localPaths[splitName]["baseDir"]
            trainPathsFile = localPaths[splitName]["trainFile"]
            valPathsFile = localPaths[splitName]["valFile"]
            testPathsFile = localPaths[splitName]["testFile"]
            comparisonPathsFile = localPaths[splitName]["comparisonFile"]
            # Train
            splitDataloaders['train'] = MTLCC24Dataloader(config, rootDir, trainPathsFile, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=True, seed = seed) # pass in paths to data
            # Validate
            splitDataloaders['validate'] = MTLCC24Dataloader(config, rootDir, valPathsFile, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle = False, seed = seed) # pass in paths to data
            # Test
            splitDataloaders['test'] = MTLCC24Dataloader(config, rootDir, testPathsFile, maskBackground, batchSize=batchSize, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data
            # Comparison
            splitDataloaders['comparison'] = MTLCC24Dataloader(config, rootDir, comparisonPathsFile, maskBackground, batchSize=1, numWorkers=workers, shuffle=False, seed = seed) # pass in paths to data

        # add in a dataframe of the class stats for if its ever needed to be used
        try:
            classStats = pd.read_csv(join(rootDir, config["LOCAL"]["classStatsFile"]))
        except:
            classStats = None

        foldLoaders.append(splitDataloaders)
    
    return foldLoaders, classStats


