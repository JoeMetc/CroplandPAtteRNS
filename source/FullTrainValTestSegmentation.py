#####
# Author: Joseph Metcalfe
# Creation Date: 28/10/2025
# Purpose: all-in-one runner for train/val/test cycle with transformer models for multispectral multitemporal Sentinel-2 data
#####

from os import path
from glob import glob
from shutil import copyfile
from pathlib import Path as PLPath
from sys import exit
from time import time
from tqdm import tqdm
from datetime import datetime
import re
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler # 'pip install timm'

from datasets import SITSDataloader
from models import loadModel
from utils.losses import CEMaskedLoss
from utils.configIO import yamlRead
from utils.visualisation import printClasswiseMetricsTable
from utils.metrics import segmentationMetrics, multiclassBoundaryIoU, flattenVises
from utils.modelUtils import EarlyStopping, pruneCheckpoints

CONFIGFILE = "C:/Users/joem8/Projects/CroplandPAtteRNS/source/configs/PAtteRNS-MTLCC-48.yaml"


def trainValidate(config, runDevice, model, dataloaders, classStats, foldNum, configFile, baseRunDir):
    # Main function for training and validating a model

    def train(model, data, optimiser, scaler, lossFn):
        # reset gradient of optimiser per batch of train data
        # optimiser.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # split data into imagery and classes
            if config["MODEL"]["dates"] == "paired": # if operating dataloader for utae etc, strip out paired dates
                idx, sitsData, dates, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                dates = dates.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                # get validation predictions
                outputs = model(sitsData, dates)
            else:
                idx, sitsData, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                # get validation predictions
                outputs = model(sitsData)

            # calc loss and gradient
            loss = lossFn(outputs, labels)
        # outside autocast
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad(set_to_none=True)

        return outputs, labels, loss

    def validate(model, data, lossFn):
        # split data into imagery and classes
        if config["MODEL"]["dates"] == "paired": # if operating dataloader for utae etc, strip out paired dates
            idx, sitsData, dates, labels = data
            sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
            dates = dates.to(runDevice, non_blocking=True, dtype=torch.float32)
            labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
            # get validation predictions
            outputs = model(sitsData, dates)
        else:
            idx, sitsData, labels = data
            sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
            labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
            # get validation predictions
            outputs = model(sitsData)

        # validation loss
        loss = lossFn(outputs, labels)

        return outputs, labels, loss


    epochs = config["RUNTIME"]["epochs"]
    saveThresh = config["LOCAL"]["saveMIoUThresh"]

    # config variables for class masking behaviour 
    maskBackground = config["MODEL"]["maskBackground"]

    # set up the datasets and data loaders for training and validation - using custom SITS datasets
    trainDataloader = dataloaders["train"]
    valDataloader = dataloaders["validate"]

    # set up optimiser and lr schedulers
    optimiser = torch.optim.Adam(model.parameters(), lr = config["RUNTIME"]["lr"], weight_decay = config["RUNTIME"]["weightDecay"], fused=True)
    optimiser.zero_grad(set_to_none=True)

    scheduler = CosineLRScheduler(
        optimiser,
        t_initial=epochs,
        lr_min=config["RUNTIME"]["lr"]/config["RUNTIME"]["lrMinDivisor"], # config["RUNTIME"]["lr"]/100
        warmup_lr_init=config["RUNTIME"]["lr"]/10000, # config["RUNTIME"]["lr"]/? 
        warmup_t=config["RUNTIME"]["warmupEpochs"],
        cycle_limit=1,
        t_in_epochs=True,
    )

    # set up early stopping check class
    earlyStopChecker = EarlyStopping(patience=config["RUNTIME"]["earlyStopPatience"], allowedMargin=config["RUNTIME"]["earlyStopMargin"], minEpochs=config["RUNTIME"]["earlyStopMinEpochs"])

    # create per-fold results dir
    modelName = config["LOCAL"]["modelName"]
    resultsDir = path.join(baseRunDir, f"fold{foldNum}")
    PLPath(resultsDir).mkdir(parents=True)

    modelPath = path.join(resultsDir, f"{modelName}_fold{foldNum}")
    epochMetricsPath = path.join(resultsDir, f"{modelName}_fold{foldNum}FullMetrics.csv")
    summaryMetricsPath = path.join(resultsDir, f"{modelName}_fold{foldNum}BestEpochSummary.csv")

    # set up metric storage list
    perEpochMetrics = []
    bestMIoU = -1
    bestMIoUVal = -1
    bestEpochMetrics = None
    bestEpochVisulisations = [None, None] # predictions, labels

    # set up loss functions
    lossFnTrain = CEMaskedLoss(config, maskBackground)
    lossFnValidate = CEMaskedLoss(config, maskBackground)

    # Grad scaler for using amp mixed precision
    scaler = torch.amp.GradScaler()

    if maskBackground:
        ignoredClasses = [0, 19]
    else:
        ignoredClasses = [19]

    print("Data Loaded, Starting Training")
    print("----------------------------------------")
    trainStartTime = time()

    # with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
    for epoch in range(1, epochs+1):
        # where the actual training happens
        print(f"Fold {foldNum}, Epoch {epoch}")

        epochStartTime = time()

        # lists to fill up with one visualisation per batch of last epoch validations, resetting each epoch
        visPredictions = []
        visLabels = []

        # run training steps for the epoch in data loader batches
        model.train(True)
        trainLosses = []
        with tqdm(trainDataloader, unit="batch") as epochTrain:
            epochTrain.set_description(f"Training progress")
            for dataBatch in epochTrain:
                
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs, labels, trainLoss = train(model, dataBatch, optimiser, scaler, lossFnTrain)
                if torch.isnan(trainLoss): # in case of pure-void tiles in PASTIS
                    continue
                trainLosses.append(trainLoss.item())

            scheduler.step(epoch)

        # start validation
        model.train(False)
        # run validation set with no gradient calculations as not needed
        with torch.no_grad():
            valLosses = []
            with tqdm(valDataloader, unit="batch") as epochValidate:
                epochValidate.set_description(f"Validate progress")
                for dataBatch in epochValidate:
                    
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs, labels, valLoss = validate(model, dataBatch, lossFnValidate)
                        if torch.isnan(valLoss): # in case of pure-void tiles in PASTIS
                            continue
                        valLosses.append(valLoss.item())
                    
                    # save segmented classes images
                    visPredictions.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
                    visLabels.append(labels.cpu().detach().numpy())


        accuracy, miou, dice, precision, recall = segmentationMetrics(flattenVises(visPredictions), flattenVises(visLabels), config["MODEL"]["classCount"], config["MODEL"]["classNames"], generateMatrix=False, ignoreClasses=ignoredClasses)
        boundaryIoU = multiclassBoundaryIoU(flattenVises(visPredictions), flattenVises(visLabels), config["MODEL"]["classCount"], ignoreClasses = ignoredClasses, dilation = 0.05)

        avgTrainLoss = np.average(trainLosses)
        avgValLoss = np.average(valLosses)
        lr = scheduler._get_values(epoch-1)[0]
        print(f"Average Train Loss: {round(avgTrainLoss, 5)}, Average Validation Loss: {round(avgValLoss, 5)}\nAccuracy: {round(accuracy, 3)}%, mIoU: {round(miou, 5)}, Val Mean Boundary IoU: {round(boundaryIoU, 5)}\nVal Dice Score: {round(dice, 5)}\nVal Precision: {round(precision, 5)}\nVal Recall: {round(recall, 5)}")
        print(f"Current Learning Rate: {lr}")

        # save an entry of the metrics for this epoch
        perEpochMetrics.append([epoch, avgTrainLoss, avgValLoss, round(accuracy, 5), round(miou, 5), round(boundaryIoU, 5), round(dice, 5), round(precision, 5), round(recall, 5), lr, int(time()-epochStartTime)])
        
        # if this epoch is current best mIoU save state and metrics
        if miou > bestMIoU:
            bestMIoU = miou
            bestMIoUVal = avgValLoss
            # only save model weights if config option on for it
            if config["LOCAL"]["saveModel"]:
                if bestMIoU >= saveThresh: # catch to reduce uncessesary saves
                    torch.save(model.state_dict(), f"{modelPath}PrimaryStateDictEPOCH{epoch}.pth")
                    print(f"Saved current best model to {modelPath}")
            # save the state dict of the best by miou to be used in test step
            bestEpochStateDict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bestEpoch = epoch
            # save the metrics for this current epoch to display at end of training
            # epoch, train loss, val loss, miou
            bestEpochMetrics = [epoch, avgTrainLoss, avgValLoss, round(accuracy, 5), round(miou, 5), round(boundaryIoU, 5), round(dice, 5), round(precision, 5), round(recall, 5), lr, int(time()-epochStartTime)]
            # save the visulisations for this epoch to display conf matrix at end
            bestEpochVisulisations[0] = visPredictions
            bestEpochVisulisations[1] = visLabels
        elif (miou >= (bestMIoU * 0.9875)) and miou >= saveThresh: # within 1.25% of best miou, save anyways
            torch.save(model.state_dict(), f"{modelPath}SecondaryStateDictEPOCH{epoch}.pth")
            print(f"Saved secondary candidate model to {modelPath}")
        elif (avgValLoss <= bestMIoUVal) and miou >= saveThresh: # better val loss than best epoch, save
            torch.save(model.state_dict(), f"{modelPath}SecondaryStateDictEPOCH{epoch}.pth")
            print(f"Saved secondary candidate model to {modelPath}")

        # save model if it is last epoch regardless
        if (epoch >= epochs):
            print(f"Saving model from final Epoch...")
            torch.save(model.state_dict(), f"{modelPath}SecondaryStateDictEPOCH{epoch}.pth")
            print(f"Saved secondary candidate model to {modelPath}")

        # end training early if continual non-improvement in miou
        if earlyStopChecker.earlyStopCheck(miou, epoch):
            print(f"Early stopping triggered at Epoch {epoch} with average validation loss of {avgValLoss}")
            print(f"Saving model from this Epoch...")
            torch.save(model.state_dict(), f"{modelPath}SecondaryStateDictEPOCH{epoch}.pth")
            print(f"Saved secondary candidate model to {modelPath}")
            break

        # end training early if manual stop file detected in results dir
        if path.exists(path.join(resultsDir, "STOP.txt")):
            print(f"Manual stop detected at Epoch {epoch} with average validation loss of {avgValLoss}")
            print(f"Saving model from this Epoch...")
            torch.save(model.state_dict(), f"{modelPath}SecondaryStateDictEPOCH{epoch}.pth")
            print(f"Saved secondary candidate model to {modelPath}")
            break

        print("----------------------------------------")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    print(f"Total run time: {time()-trainStartTime:.0f}s")

    metricColumns = ["Epoch", "Training Loss", "Validation Loss", "Accuracy", "mIoU", "mBIoU", "Dice Score", "Precision", "Recall", "EpochEndLR", "Duration (s)"]
    metricLog = pd.DataFrame(perEpochMetrics, columns = metricColumns)
    bestEpochLog = pd.DataFrame([bestEpochMetrics], columns = metricColumns)
    metricLog.set_index("Epoch", inplace=True)
    bestEpochLog.set_index("Epoch", inplace=True)

    metricLog.to_csv(epochMetricsPath)
    bestEpochLog.to_csv(summaryMetricsPath)
    print(f"Saved logs to {epochMetricsPath}")

    # save the preds, labels, and conf matrix numerically (and CM visually)
    segmentationMetrics(flattenVises(bestEpochVisulisations[0]), flattenVises(bestEpochVisulisations[1]), config["MODEL"]["classCount"], config["MODEL"]["classNames"], generateMatrix=True, ignoreClasses=ignoredClasses, saveDir = resultsDir, filePrefix="bestValEpoch")

    return resultsDir, model, bestEpochStateDict, bestEpoch


def runInference(config, runDevice, model, dataloaders, classStats, foldNum, resultsDir):

    model.eval()

    maskBackground = config["MODEL"]["maskBackground"]

    if maskBackground:
        ignoredClasses = [0, 19]
    else:
        ignoredClasses = [19]

    testDataloader = dataloaders['test']

    testLosses = []
    visPredictions = []
    visLabels = []

    lossFnTest = CEMaskedLoss(config, maskBackground)

    reportInterval = 500
    reportStep = 0

    with torch.no_grad():
        for index, data in enumerate(testDataloader):

            if reportStep >= reportInterval:
                print(f"Running Batch {index}")
                reportStep = 0
            else:
                reportStep += 1

            if config["MODEL"]["dates"] == "paired": # utae
                idx, sitsData, dates, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                dates = dates.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                outputs = model(sitsData, dates)
            else:
                idx, sitsData, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                outputs = model(sitsData)

            loss = lossFnTest(outputs, labels)
            testLosses.append(loss.cpu().detach().numpy())

            visPredictions.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
            visLabels.append(labels.cpu().detach().numpy())

    avgTestLoss = np.average(testLosses)

    flatPreds = flattenVises(visPredictions)
    flatLabels = flattenVises(visLabels)

    accuracy, miou, dice, precision, recall = segmentationMetrics(
        flatPreds, flatLabels,
        config["MODEL"]["classCount"], config["MODEL"]["classNames"],
        generateMatrix=False, ignoreClasses=ignoredClasses
    )
    boundaryIoU = multiclassBoundaryIoU(
        flatPreds, flatLabels,
        config["MODEL"]["classCount"], ignoreClasses=ignoredClasses, dilation=0.05
    )

    print(f"Test Cycle Complete for {config['LOCAL']['modelName']} fold {foldNum}")
    print(f"Avg Test Loss: {avgTestLoss}\nTest Accuracy: {round(accuracy, 5)}%\nTest mIoU: {round(miou, 5)}\nTest Mean Boundary IoU: {round(boundaryIoU, 5)}\nTest Dice Score: {round(dice, 5)}\nTest Precision: {round(precision, 5)}\nTest Recall: {round(recall, 5)}")

    return {
        "Test Loss": float(avgTestLoss),
        "Accuracy": round(accuracy, 5),
        "mIoU": round(miou, 5),
        "mBIoU": round(boundaryIoU, 5),
        "Dice Score": round(dice, 5),
        "Precision": round(precision, 5),
        "Recall": round(recall, 5)
    }
    
def runSequentialInference(config, runDevice, dataloaders, classStats, foldNum, resultsDir):

    allMetrics = []
    rankedCheckpoints = []
    bestMIoU = -1
    bestPthPath = None

    for pthPath in sorted(glob(path.join(resultsDir, "*.pth")), key=path.getmtime):
        _, fname = path.split(pthPath)
        epochMatch = re.search(r"EPOCH(\d+)", fname)
        epoch = int(epochMatch.group(1)) if epochMatch else -1
        modelType = "Primary" if "Primary" in fname else "Secondary" if "Secondary" in fname else "Unknown"

        print(f"Testing {fname}")

        seqModel = loadModel(config, runDevice)
        seqModel.load_state_dict(torch.load(pthPath, weights_only=True))

        metrics = runInference(config, runDevice, seqModel, dataloaders, classStats, foldNum, resultsDir)

        del seqModel
        with torch.no_grad():
            torch.cuda.empty_cache()

        if metrics["mIoU"] > bestMIoU:
            bestMIoU = metrics["mIoU"]
            bestPthPath = pthPath

        allMetrics.append({"Epoch": epoch, "Type": modelType, **metrics})
        rankedCheckpoints.append((pthPath, metrics["mIoU"]))

    df = pd.DataFrame(allMetrics).sort_values("Epoch")
    seqCsvPath = path.join(resultsDir, f"{config['LOCAL']['modelName']}_fold{foldNum}_SequentialTestMetrics.csv")
    df.to_csv(seqCsvPath, index=False)
    print(f"Saved sequential test metrics to {seqCsvPath}")

    print(f"Best checkpoint by mIoU: {path.split(bestPthPath)[1]} ({bestMIoU})")
    saveBestEpochDiagnostics(config, runDevice, bestPthPath, dataloaders, resultsDir)

    bestRow = df.loc[df["mIoU"].idxmax()].to_dict()

    if args.keep is not None:
        pruneCheckpoints(rankedCheckpoints, foldNum, args.keep)

    return bestRow

def saveBestEpochDiagnostics(config, runDevice, bestPthPath, dataloaders, resultsDir):

    maskBackground = config["MODEL"]["maskBackground"]

    if maskBackground:
        ignoredClasses = [0, 19]
    else:
        ignoredClasses = [19]

    testDataloader = dataloaders['test']

    visPredictions = []
    visLabels = []

    bestModel = loadModel(config, runDevice)
    bestModel.load_state_dict(torch.load(bestPthPath, weights_only=True))
    bestModel.eval()

    with torch.no_grad():
        for data in testDataloader:
            if config["MODEL"]["dates"] == "paired":
                idx, sitsData, dates, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                dates = dates.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                outputs = bestModel(sitsData, dates)
            else:
                idx, sitsData, labels = data
                sitsData = sitsData.to(runDevice, non_blocking=True, dtype=torch.float32)
                labels = labels.to(runDevice, non_blocking=True, dtype=torch.long)
                outputs = bestModel(sitsData)

            visPredictions.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
            visLabels.append(labels.cpu().detach().numpy())

    del bestModel
    with torch.no_grad():
        torch.cuda.empty_cache()

    flatPreds = flattenVises(visPredictions)
    flatLabels = flattenVises(visLabels)

    # confusion matrix
    segmentationMetrics(
        flatPreds, flatLabels,
        config["MODEL"]["classCount"], config["MODEL"]["classNames"],
        generateMatrix=True, ignoreClasses=ignoredClasses,
        saveDir=resultsDir, filePrefix="testSet"
    )

    # classwise metrics
    cm, _, cw_miou, cw_dice, cw_precision, cw_recall = segmentationMetrics(
        flatPreds, flatLabels,
        config["MODEL"]["classCount"], config["MODEL"]["classNames"],
        generateMatrix=False, ignoreClasses=ignoredClasses,
        returnCM=True, classwiseMetrics=True
    )
    printClasswiseMetricsTable(
        config["MODEL"]["classNames"],
        config["MODEL"]["classCount"],
        ignoredClasses,
        cw_miou,
        cw_dice,
        cw_precision,
        cw_recall,
        saveCSV=True,
        CSVPath=resultsDir
    )

    print(f"Saved confusion matrix and classwise metrics for best epoch to {resultsDir}")

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
    print(f"GPU Registered: {torch.cuda.is_available()} @ GPU {deviceID}")
    return device

def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Main Program
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--keep", type=int, default=None,
    help="None keeps all checkpoint .pths. If set, keep only this many of the top-mIoU checkpoints per fold after sequential testing, plus the worst as a floor. E.g. --keep 5 retains the 5 best and the worst, deleting everything else.")
    args = parser.parse_args()

    configFile = args.config if args.config is not None else CONFIGFILE
    print(configFile)

    config = yamlRead(configFile)
    # Use GPU if available
    runDevice = setRunDevice(config)

    runDate = datetime.today().strftime('%d-%m-%Y')
    architecture = config["MODEL"]["architecture"]
    modelName = config["LOCAL"]["modelName"]
    resultsParentDir = config["LOCAL"]["resultsDir"]

    baseRunDir = path.join(resultsParentDir, f"{architecture}_{modelName}_{runDate}")

    fileNotMade = True
    try:
        PLPath(baseRunDir).mkdir(parents=True)
    except FileExistsError:
        incrementer = 0
        while fileNotMade:
            try:
                baseRunDir = path.join(resultsParentDir, f"{architecture}_{modelName}_{runDate}-{incrementer}")
                PLPath(baseRunDir).mkdir(parents=True)
                print(f"Model name already in use - auto incrementing model name to {modelName}-{incrementer}")
                fileNotMade = False
            except FileExistsError:
                incrementer += 1

    # save a copy of config to results dir for later reference
    copyfile(configFile, path.join(baseRunDir, f"{modelName}_config.yaml"))

    if config["LOCAL"]["seed"] is not None:
        seed = config["LOCAL"]["seed"]
    else:
        seed = 1 # 1 is defualt
    setSeed(seed)
    
    dataloaders, classStats = SITSDataloader(config, seed)

    allFoldBestMetrics = []

    for fold in range(len(dataloaders)):
        model = loadModel(config, runDevice)

        foldResultsDir, _, bestEpochStateDict, bestEpoch = trainValidate(
            config, runDevice, model, dataloaders[fold], classStats, fold+1, configFile, baseRunDir
        )

        del _
        with torch.no_grad():
            torch.cuda.empty_cache()

        # Run sequential inference across all saved checkpoints for this fold
        bestFoldMetrics = runSequentialInference(
            config, runDevice, dataloaders[fold], classStats, fold+1, foldResultsDir
        )
        bestFoldMetrics["Fold"] = fold + 1
        allFoldBestMetrics.append(bestFoldMetrics)

        del model, bestEpochStateDict
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(runDevice)

        for loader in dataloaders[fold].values():
            loader._iterator = None  # force worker shutdown
        dataloaders[fold] = None

    # Per-fold best results CSV
    foldSummaryDf = pd.DataFrame(allFoldBestMetrics)
    foldSummaryDf.to_csv(path.join(baseRunDir, f"{modelName}_AllFoldsBestEpoch.csv"), index=False)

    # Aggregate statistics CSV
    metricCols = ["Test Loss", "Accuracy", "mIoU", "mBIoU", "Dice Score", "Precision", "Recall"]
    aggRows = []
    for col in metricCols:
        vals = foldSummaryDf[col]
        aggRows.append({
            "Metric": col,
            "Mean": round(vals.mean(), 5),
            "Median": round(vals.median(), 5),
            "Std Dev": round(vals.std(), 5),
            "Range": round(vals.max() - vals.min(), 5),
            "Min": round(vals.min(), 5),
            "Max": round(vals.max(), 5),
        })
    aggDf = pd.DataFrame(aggRows)
    aggDf.to_csv(path.join(baseRunDir, f"{modelName}_AllFoldsAggregate.csv"), index=False)
    print(f"Saved cross-fold summaries to {baseRunDir}")

    gc.collect()

    



    
    
