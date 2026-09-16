#####
# Author: Joseph Metcalfe
# Creation Date: 16/05/2025
# Purpose: Reporting confusion matrices and metric scores for crop class predictions and labels
#####

import torch
import numpy as np
from os import path
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
import matplotlib.pyplot as plt
import seaborn


def flattenVises(visList):
    return [vis for batchVis in visList for vis in batchVis]

def calcMIoU(confMatrix, numClasses, classwise=False):
    # calc IoUs per class, IoU = TP/TP+FP+FN
    IoUs = []
    for c in range(numClasses):
        tp = float(confMatrix[c, c])
        fn = float(np.sum(confMatrix[:, c]) - tp)
        fp = float(np.sum(confMatrix[c, :]) - tp)
        IoUs.append(tp / (tp + fp + fn + 1e-16)) # avoid dbz

    if classwise:
        return IoUs
    return np.mean(IoUs) # macro

def calcPrecisionRecall(confMatrix, numClasses, classwise=False):
    precisions = []
    recalls = []

    for c in range(numClasses):
        tp = float(confMatrix[c, c])
        fn = float(np.sum(confMatrix[c, :]) - tp)
        fp = float(np.sum(confMatrix[:, c]) - tp)

        precision = tp / (tp + fp + 1.0e-16)
        recall = tp / (tp + fn + 1.0e-16)

        precisions.append(precision)
        recalls.append(recall)

    if classwise:
        return precisions, recalls

    return np.mean(precisions), np.mean(recalls)

def calcAccuracy(confMatrix): # OA%
    correct = np.trace(confMatrix) # sum of true positives across all classes
    total = np.sum(confMatrix)# total pixels
    accuracy = (correct / (total + 1e-16)) * 100
    return accuracy

def calcDice(confMatrix, numClasses, classwise=False, epsilon=1e-6):
    """Calculates mean Dice coefficient (%) across all classes from confusion matrix."""
    dice_scores = []
    for c in range(numClasses):
        tp = float(confMatrix[c, c])
        fp = float(np.sum(confMatrix[c, :]) - tp)
        fn = float(np.sum(confMatrix[:, c]) - tp)

        denom = (2 * tp + fp + fn + epsilon)
        if denom > 0:
            dice = (2 * tp) / denom
            dice_scores.append(dice)

    if classwise:
        return dice_scores
    return np.mean(dice_scores)

def segmentationMetrics(predictions, labels, numClasses, classNamesList, generateMatrix = True, ignoreClasses = [], saveDir = None, filePrefix = "", returnCM = False, classwiseMetrics=False):
    """Calculates MIoU & Accuracy for segmentation results"""
    classIndecies = list(range(0, numClasses))
    for ignoredClass in ignoreClasses:
        try:
            classIndecies.remove(ignoredClass)
            numClasses -= 1
        except:
            """"""
    classNamesList = [classNamesList[i] for i in classIndecies]

    # string everything out in a 1D array as its just matching positions up anyways
    preds = np.concatenate(predictions)
    labels = np.concatenate(labels)

   # mask ignored indexes present in the labels
    mask = (~np.isin(labels, ignoreClasses)) & (~np.isin(preds, ignoreClasses))
    preds = preds[mask]
    labels = labels[mask]

    # ensure only valid classes remain
    validClassSet = set(classIndecies)
    validMask = np.isin(preds, list(validClassSet)) & np.isin(labels, list(validClassSet))

    preds = preds[validMask]
    labels = labels[validMask]

    # reindex
    old_to_new = {old: new for new, old in enumerate(classIndecies)}
    preds = np.vectorize(old_to_new.get)(preds)
    labels = np.vectorize(old_to_new.get)(labels)

    indices = labels * numClasses + preds
    confMatrix = np.bincount(indices, minlength=numClasses**2).reshape((numClasses, numClasses))

    # added to remove 0 population samples from averages
    gt_pixel_counts = confMatrix.sum(axis=1)  # sum over predictions
    valid_classes = gt_pixel_counts > 0
    confMatrix = confMatrix[valid_classes][:, valid_classes]
    numClasses = confMatrix.shape[0]

    # accuracy % (not classwise)
    accuracy = calcAccuracy(confMatrix)

    # mIoU
    mIoU = calcMIoU(confMatrix, numClasses, classwise=classwiseMetrics)

    # Precision & Recall
    precision, recall = calcPrecisionRecall(confMatrix, numClasses, classwise=classwiseMetrics)

    # Dice
    dice = calcDice(confMatrix, numClasses, classwise=classwiseMetrics)

    if generateMatrix:
        # create percentage-wise matrix
        labelSums = confMatrix.sum(axis=1, keepdims=True)
        confMatrixPercentages = np.divide(
        confMatrix, labelSums, out=np.zeros_like(confMatrix, dtype=float), where=labelSums!=0) * 100

        # plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(11,11))
        axs = plt.subplot()
        seaborn.heatmap(confMatrixPercentages, annot = True, cmap = "Blues", vmin = 0, vmax = 100, fmt = ".1f", cbar = True, square = True, ax = axs)
        axs.set_ylabel('Actual Class', weight = 'bold')
        axs.set_xlabel('Predicted Class', weight = 'bold')
        labels = list(range(0,len(classNamesList)))
        labels = [label + 0.5 for label in labels]
        plt.xticks(labels, classNamesList, weight = 'bold', rotation = 90)
        plt.yticks(labels, classNamesList, weight = 'bold', rotation = 0)
        plt.title("Actual Class vs Predicted Class Excluding Void Parcels")
        plt.subplots_adjust(left=0.2, right=1, top=0.9, bottom=0.2)

        # if a results dir is given, save there instead of bringing up in GUI
        if saveDir == None:
            plt.show()
        else:
            plt.savefig(path.join(saveDir,f"{filePrefix}visualCM.png"))
            np.save(path.join(saveDir,f"{filePrefix}cm"), confMatrix)

    if returnCM:
        return confMatrix, accuracy, mIoU, dice, precision, recall
    return accuracy, mIoU, dice, precision, recall

def multiclassBoundaryIoU(predictions, labels, numClasses, ignoreClasses=[], dilation=0.05):
    """
    Returns mean Boundary IoU calculated over all classes individually then meaned,
    excluding ignored classes and classes absent from both predictions and labels.

    Dilation given as a float for a % dilation from boundaries

    Derived from https://arxiv.org/abs/2103.16562
    """
    classIndecies = [c for c in range(numClasses) if c not in ignoreClasses]

    # all arrays stay in original index space (0..numClasses-1)
    boundaryIoUArray = np.full((len(predictions), numClasses), np.nan)
    tiles = list(zip(predictions, labels))

    # track which classes actually appear in the data (ignoring ignored ones)
    presentClasses = np.zeros(numClasses, dtype=bool)
    for preds, lbls in tiles:
        unique_classes = np.unique(np.concatenate((preds.flatten(), lbls.flatten())))
        unique_classes = unique_classes[~np.isin(unique_classes, ignoreClasses)]
        unique_classes = unique_classes[(unique_classes >= 0) & (unique_classes < numClasses)]
        presentClasses[unique_classes] = True

    for tileIdx in range(len(predictions)):
        h, w = tiles[tileIdx][1].shape
        dilationPixels = max(1, round(dilation * np.sqrt(h**2 + w**2)))

        ignoreMask = np.isin(tiles[tileIdx][1], ignoreClasses)
        ignoreMaskDilated = binary_dilation(ignoreMask, iterations=dilationPixels)

        for classIdx in classIndecies:
            perClassMaskPreds = (tiles[tileIdx][0] == classIdx)
            erodedPreds = binary_erosion(perClassMaskPreds, border_value=1)
            boundaryPreds = perClassMaskPreds & ~erodedPreds
            boundaryPredsMasked = boundaryPreds & ~ignoreMaskDilated
            dilatedPreds = ~binary_erosion(~boundaryPredsMasked, iterations=dilationPixels, border_value=1)
            dilatedPreds = dilatedPreds & perClassMaskPreds  # (Pd ∩ P)

            perClassMaskLabels = (tiles[tileIdx][1] == classIdx)
            erodedLabels = binary_erosion(perClassMaskLabels, border_value=1)
            boundaryLabels = perClassMaskLabels & ~erodedLabels
            boundaryLabelsMasked = boundaryLabels & ~ignoreMaskDilated
            dilatedLabels = ~binary_erosion(~boundaryLabelsMasked, iterations=dilationPixels, border_value=1)
            dilatedLabels = dilatedLabels & perClassMaskLabels  # (Gd ∩ G)

            boundaryIntersection = dilatedLabels & dilatedPreds
            boundaryUnion = dilatedLabels | dilatedPreds

            if np.sum(boundaryIntersection) + np.sum(boundaryUnion) > 0:
                boundaryIoUArray[tileIdx][classIdx] = (
                    np.sum(boundaryIntersection) / (np.sum(boundaryUnion) + 1e-8)
                )
            # else stays nan

    # average over tiles per class
    perClassBoundaryIoU = np.nanmean(boundaryIoUArray, axis=0)

    # valid = not ignored AND present in data
    ignoreSet = np.zeros(numClasses, dtype=bool)
    ignoreSet[[c for c in ignoreClasses if 0 <= c < numClasses]] = True
    valid_mask = ~ignoreSet & presentClasses

    return np.nanmean(perClassBoundaryIoU[valid_mask])




class InferenceTimer:
    def __init__(self, warmup_steps=10): # discard first instances to let proper cached data be used / avoid hardware bias
        self.warmup_steps = warmup_steps
        self._step = 0
        self._times = []
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._start.record()

    def stop(self):
        self._end.record()
        torch.cuda.synchronize()
        elapsed = self._start.elapsed_time(self._end)
        if self._step >= self.warmup_steps:
            self._times.append(elapsed)
        self._step += 1

    def summary(self):
        if not self._times:
            return "No measurements recorded"
        t = torch.tensor(self._times)
        return {
            "mean_ms": t.mean().item(),
            "std_ms": t.std().item(),
            "min_ms": t.min().item(),
            "max_ms": t.max().item(),
            "n": len(self._times),
        }
