#####
# Author: Joseph Metcalfe
# Creation Date: 17/03/2025
# Purpose: Simple inference runner for temporal segmentation models on the test set of config dataset
#####

from os import path
import numpy as np
import torch
import torch.nn as nn

from datasets import SITSDataloader
from models import loadModel
from utils.losses import CEMaskedLoss
from utils.configIO import yamlRead
from utils.visualisation import viewSideBySide, printClasswiseMetricsTable, viewBoundaryError
from utils.metrics import segmentationMetrics, multiclassBoundaryIoU, flattenVises, InferenceTimer


CONFIGFILE = "source/configs/PAtteRNS-PASTIS-128.yaml"
MODELPATHS = "source/datasets/pathconfs/savedModels.yaml"
USEDMODEL = "BoundryErrVisPASTIS128PAtteRNSP4Seed1"  # for multiseed, should point at the parent dir containing one subdir per seed, otherwise single .pth file
STATEDICT = True
VIS = False
CLASSWISEMETRICS = False
SAVEMETRICS = False


def runInference(config, runDevice, model, dataloaders, classStats, modelPath):
    
    timer = InferenceTimer(warmup_steps=10)

    # set model to only evaluate instead of learning
    model.eval()

    # config variables for class masking behaviour
    maskBackground = config["MODEL"]["maskBackground"]

    # load in the test set
    testDataloader = dataloaders['test']
    print("Data Loaded")

    # initialise variables for data and metrics held across all batches 
    testLosses = []
    visPredictions = []
    visLabels = []

    # set up loss function
    lossFnTest = CEMaskedLoss(config, maskBackground)

    # how often it prints as going through batches, just to track progress
    reportInterval = 500
    reportStep = 0

    # run through the dataset
    with torch.no_grad():
        for index, data in enumerate(testDataloader):

            if reportStep >= reportInterval:
                print(f"Running Batch {index}")
                reportStep = 0
            else:
                reportStep += 1

            if config["MODEL"]["dates"] == "paired": # if operating dataloader for utae etc, strip out paired dates
                idx, sitsData, dates, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                dates = dates.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                timer.start()
                outputs = model(sitsData, dates)
                timer.stop()
            else:
                idx, sitsData, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                timer.start()
                outputs = model(sitsData)
                timer.stop()

            # calc test loss
            loss = lossFnTest(outputs, labels)
            testLosses.append(loss.cpu().detach().numpy())

            # save segmented classes images
            visPredictions.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
            visLabels.append(labels.cpu().detach().numpy())

    # inference speed per batch
    print(timer.summary())
        
    # average out some metrics across all test images
    avgTestLoss = np.average(testLosses)

    if maskBackground:
        ignoredClasses = [0, 19]
    else:
        ignoredClasses = [19]
    
    accuracy, miou, dice, precision, recall = segmentationMetrics(flattenVises(visPredictions), flattenVises(visLabels), config["MODEL"]["classCount"], config["MODEL"]["classNames"], generateMatrix=False, ignoreClasses=ignoredClasses)
    boundaryIoU = multiclassBoundaryIoU(flattenVises(visPredictions), flattenVises(visLabels), config["MODEL"]["classCount"], ignoreClasses = ignoredClasses, dilation = 0.05)

    print(f"Test Cycle Complete for {USEDMODEL}")
    print(f"Avg Test Loss: {avgTestLoss}\nTest Accuracy: {round(accuracy, 5)}%\nTest mIoU: {round(miou, 5)}\nTest Mean Boundary IoU: {round(boundaryIoU, 5)}\nTest Dice Score: {round(dice, 5)}\nTest Precision: {round(precision, 5)}\nTest Recall: {round(recall, 5)}")

    if CLASSWISEMETRICS:
        cm, accuracy, miou, dice, precision, recall = segmentationMetrics(flattenVises(visPredictions), flattenVises(visLabels), config["MODEL"]["classCount"], config["MODEL"]["classNames"], generateMatrix=False, ignoreClasses=ignoredClasses, returnCM=True, classwiseMetrics=True)

        modelDir = path.split(modelPath)[0]

        printClasswiseMetricsTable(config["MODEL"]["classNames"], config["MODEL"]["classCount"], ignoredClasses, miou, dice, precision, recall, SAVEMETRICS, modelDir)   

    if VIS:
        # one per batch visualisations
        for batch in range(0,len(visPredictions)):
            viewSideBySide(visLabels[batch][0], visPredictions[batch][0])
            # viewBoundaryError(visLabels[batch][0], visPredictions[batch][0], showLabels=True, skipShow=True,
            #                   saveLabelsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC48Fold0Labels/batch_{batch}",
            #                   saveLegendPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC48Fold0Legends/batch_{batch}",
            #                   savePredsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC48Fold0Outs/UNet3D/batch_{batch}",
            #                   saveBoundaryErrPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC48Fold0Errs/UNet3D/batch_{batch}"
            #                 )
            # viewBoundaryError(visLabels[batch][0], visPredictions[batch][0], showLabels=True, skipShow=True,
            #                   saveLabelsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC24Fold0Labels/batch_{batch}",
            #                   saveLegendPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC24Fold0Legends/batch_{batch}",
            #                   savePredsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC24Fold0Outs/PAtteRNSP4/batch_{batch}",
            #                   saveBoundaryErrPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/MTLCC24Fold0Errs/PAtteRNSP4/batch_{batch}"
            #                 )
            # viewBoundaryError(visLabels[batch][0], visPredictions[batch][0], showLabels=True, skipShow=True,
            #                 #   saveLabelsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS24Fold1Labels/batch_{batch}",
            #                 #   saveLegendPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS24Fold1Legends/batch_{batch}",
            #                 #   savePredsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS24Fold1Outs/UNet3D/batch_{batch}",
            #                 #   saveBoundaryErrPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS24Fold1Errs/UNet3D/batch_{batch}"
            #                 )                
            # viewBoundaryError(visLabels[batch][0], visPredictions[batch][0], showLabels=True, skipShow=True,
            #                 #   saveLabelsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS128Fold1Labels/batch_{batch}",
            #                 #   saveLegendPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS128Fold1Legends/batch_{batch}",
            #                 #   savePredsPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS128Fold1Outs/PAtteRNSP4/batch_{batch}",
            #                 #   saveBoundaryErrPath=f"D:/Models/TemporalSegmentation/Results/Visualisations/BoundaryErrors/PASTIS128Fold1Errs/PAtteRNSP4/batch_{batch}"
            #                 )
            # PAtteRNSP2 PAtteRNSP4 TSViTP2 TSViTP4 UTAE UNet3D

def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
            
def setRunDevice(config, cpu=False):
    dynamicGPU = config["LOCAL"]["dynamicGPU"]
    if dynamicGPU:
        deviceID = torch.cuda.current_device()
    else:
        deviceID = config["LOCAL"]["gpuNum"]
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{deviceID}")
    elif cpu:
        device = torch.device("cpu")
    else:
        exit("No available runtime hardware")
    if not cpu:
        print(f"GPU Registered: {torch.cuda.is_available()} for GPU {deviceID}")
    return device

if __name__ == "__main__":
    # Main Program

    config = yamlRead(CONFIGFILE)
    # Use GPU if available
    runDevice = setRunDevice(config)

    modelPaths = yamlRead(MODELPATHS)

    print(modelPaths[USEDMODEL])

    if STATEDICT: # capable of taking both usual weights only state dict .pth file but also whole model files if needed
        model = loadModel(config, runDevice)
        model.load_state_dict(torch.load(modelPaths[USEDMODEL], weights_only=True))
    else:
        model = torch.load(modelPaths[USEDMODEL], weights_only=False)

    print(f"Running model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    if config["LOCAL"]["seed"] is not None:
        seed = config["LOCAL"]["seed"]
    else:
        seed = 1 # 1 is defualt
    setSeed(seed)
    
    dataloaders, classStats = SITSDataloader(config, seed)

    runInference(config, runDevice, model, dataloaders[0], classStats, modelPaths[USEDMODEL])