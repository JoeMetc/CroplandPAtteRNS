#####
# Author: Joseph Metcalfe
# Creation Date: 24/04/2025
# Purpose: Visualising segmentation results or label images from the PASTIS dataset
#####

from os import path
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_dilation, generate_binary_structure


# create a colour map to match the one used by https://github.com/VSainteuf/pastis-benchmark/blob/main/documentation/pastis-documentation.pdf for class labels
pastisColours = np.zeros(((20,4)))
pastisColours[0] = [0, 0, 0, 1]
pastisColours[1] = [0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1]
pastisColours[2] = [1.0, 0.4980392156862745, 0.054901960784313725, 1]
pastisColours[3] = [1.0, 0.7333333333333333, 0.47058823529411764, 1]
pastisColours[4] = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1]
pastisColours[5] = [0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1]
pastisColours[6] = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1]
pastisColours[7] = [1.0, 0.596078431372549, 0.5882352941176471, 1]
pastisColours[8] = [0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1]
pastisColours[9] = [0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1]
pastisColours[10] = [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1]
pastisColours[11] = [0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1]
pastisColours[12] = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1]
pastisColours[13] = [0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1]
pastisColours[14] = [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1]
pastisColours[15] = [0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1]
pastisColours[16] = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1]
pastisColours[17] = [0.8588235294117647, 0.8588235294117647, 0.5529411764705883, 1]
pastisColours[18] = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1]
pastisColours[19] = [1, 1, 1, 1]
cmapPastis = ListedColormap(pastisColours)
pastisClasses = ["Background", "Meadow", "Winter Soft Wheat", "Maize", "Winter Barley", "Winter Rapeseed", "Spring Barley", "Sunflower", "Grapevine", "Sugar Beet", "Winter Triticale", "Winter Durum Wheat", "Fruits, Vegtables, & Flowers", "Potatoes", "Leguminous Fodder", "Soybeans", "Orchard", "Mixed Cereal", "Sorghum", "Void Parcel"]

# # create a colour map to colour match identical crops in the one used by https://github.com/VSainteuf/pastis-benchmark/blob/main/documentation/pastis-documentation.pdf 
# # non overlapping crops should get a colour not in the PASTIS colours
mtlccColours = np.zeros(((18,4)))
mtlccColours[0]  = [0, 0, 0, 1]  # Unknown
mtlccColours[1]  = [0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1]  # Sugar Beet
mtlccColours[2]  = to_rgba("#e3b06e")  # Summer Oat
mtlccColours[3]  = [0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1]  # Meadow
mtlccColours[4]  = [0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1]  # Rape
mtlccColours[5]  = to_rgba("#902e62")  # Hop
mtlccColours[6]  = to_rgba("#dbcd0e")  # Winter Spelt
mtlccColours[7]  = [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1]  # Winter Triticale
mtlccColours[8]  = to_rgba("#106f89")  # Beans
mtlccColours[9]  = to_rgba("#7ee8ae")  # Peas
mtlccColours[10] = [0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1]  # Potatoe
mtlccColours[11] = [0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1]  # Soybeans
mtlccColours[12] = to_rgba("#235825")  # Asparagus
mtlccColours[13] = [1.0, 0.4980392156862745, 0.054901960784313725, 1]  # Winter Wheat
mtlccColours[14] = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1]  # Winter Barley
mtlccColours[15] = to_rgba("#f49f27")  # Winter Rye
mtlccColours[16] = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1]  # Summer Barley
mtlccColours[17] = [1.0, 0.7333333333333333, 0.47058823529411764, 1]  # Maize
cmapMTLCC = ListedColormap(mtlccColours)
mtlccClasses = ["Background", "Sugar Beet", "Summer Oat", "Meadow", "Rapeseed", "Hop", "Winter Spelt", "Winter Triticale", "Beans", "Peas", "Potatoes", "Soybeans", "Asparagus", "Winter Wheat", "Winter Barley", "Winter Rye", "Summer Barley", "Maize"]
# mtlccClasses = {0: "Background",1: "Sugar Beet",2: "Summer Oat",3: "Meadow",4: "Rapeseed",5: "Hop",6: "Winter Spelt",7: "Winter Triticale",8: "Beans",9: "Peas",10: "Potatoes",11: "Soybeans",12: "Asparagus",13: "Winter Wheat",14: "Winter Barley",15: "Winter Rye",16: "Summer Barley",17: "Maize"}

norm = BoundaryNorm(boundaries=np.arange(21)-0.5, ncolors=20)

def viewClassImage(inputClassImage):
    """Function expects a 2D np array with discrete integer class values"""
    # split figure into image and legend
    fig = plt.figure(figsize=(8,6))
    grid = GridSpec(1, 2, width_ratios=[4, 1])  # 4/5 for the image, 1/5 for the legend
    axImg = fig.add_subplot(grid[0])
    axLeg = fig.add_subplot(grid[1])
    # add image in using pastis cmap
    axImg.imshow(inputClassImage, cmap=cmapPastis, vmin=0, vmax=len(cmapPastis.colors)-1, interpolation="none")
    # create a patch label for every color 
    patches = [mpatches.Patch(facecolor=pastisColours[i], label=pastisClasses[i].format(l=pastisClasses[i]), edgecolor="Black") for i in range(len(pastisClasses))]
    # put those patched as legend-handles into the legend section of the grid
    axLeg.axis("off")
    axLeg.legend(handles=patches, bbox_to_anchor=(-0.4, 0.975), loc=2, borderaxespad=0., fontsize=9)
    plt.show()

def viewSideBySide(groundTruthImage, segmentationResultsImage):
    """Function expects two 2D np arrays with discrete integer class values, first for labels, second for outputs"""
    # split figure into image and legend
    fig = plt.figure(figsize=(10,4))
    grid = GridSpec(1, 3, width_ratios=[4, 4, 2])
    axLableImg = fig.add_subplot(grid[0])
    axOutputImg = fig.add_subplot(grid[1])
    axLeg = fig.add_subplot(grid[2])
    # add labels image in using pastis cmap
    axLableImg.imshow(groundTruthImage, cmap=cmapPastis, vmin=0, vmax=len(cmapPastis.colors)-1, interpolation="none")
    # add outputs image in using pastis cmap
    axOutputImg.imshow(segmentationResultsImage, cmap=cmapPastis, vmin=0, vmax=len(cmapPastis.colors)-1, interpolation="none")
    # create a patch label for every color 
    patches = [mpatches.Patch(facecolor=pastisColours[i], label=pastisClasses[i].format(l=pastisClasses[i]), edgecolor="Black") for i in range(len(pastisClasses))]
    # put those patched as legend-handles into the legend section of the grid
    axLeg.axis("off")
    axLeg.legend(handles=patches, bbox_to_anchor=(-0.2, 1), loc=2, borderaxespad=0., fontsize=8)
    plt.show()

def plot_class_population(ax, labelImage, classNames, classColours, fontsize=10):
    ax.clear()

    # count pixels per class
    unique, counts = np.unique(labelImage, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # keep only classes present
    classes = sorted(class_counts.keys())
    values = [class_counts[c] for c in classes]
    names = [classNames[c] for c in classes]
    colours = [classColours[c] for c in classes]

    # horizontal bar plot
    y = np.arange(len(classes))
    ax.barh(y, values, color=colours, edgecolor="black")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=fontsize)
    ax.invert_yaxis()  # top = largest index visually consistent
    ax.set_xlabel("Pixel count", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.set_title("Class population", fontsize=fontsize + 1)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def printClasswiseMetricsTable(classNames, numClasses, ignoreClasses, miou, dice, precision, recall, saveCSV = False, CSVPath = ""):
    """
    Prints a table of per class seg metrics
    """
    classIndecies = [c for c in range(numClasses) if c not in ignoreClasses]
    numClasses = len(classIndecies)
    classNames = [classNames[i] for i in classIndecies]

    print("\nClasswise Segmentation Metrics:\n")
    header = f"{'Class':<20} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8}"
    print(header)
    print("-" * len(header))

    rows = []

    for i, className in enumerate(classNames):
        iou_val  = miou[i]
        dice_val = dice[i]
        prec_val = precision[i]
        rec_val  = recall[i]

        def fmt(x):
            return "  nan" if np.isnan(x) else f"{x:6.3f}"

        print(f"{className:<20} "
            f"{fmt(iou_val):>8} "
            f"{fmt(dice_val):>8} "
            f"{fmt(prec_val):>10} "
            f"{fmt(rec_val):>8}"
        )

        rows.append([className,
            "" if np.isnan(iou_val)  else float(iou_val),
            "" if np.isnan(dice_val) else float(dice_val),
            "" if np.isnan(prec_val) else float(prec_val),
            "" if np.isnan(rec_val)  else float(rec_val),
        ])

    if saveCSV:
        csvFile = path.join(CSVPath, "InferenceClasswise.csv")

        with open(csvFile, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "IoU", "Dice", "Precision", "Recall"])
            writer.writerows(rows)

        print(f"\nSaved classwise metrics to: {csvFile}")



def computeBoundaryMask(classImage):
    """Returns a boolean array, True where a pixel differs from a
    horizontal or vertical neighbour (i.e. sits on a class boundary)."""
    boundary = np.zeros_like(classImage, dtype=bool)
    boundary[:, :-1] |= classImage[:, :-1] != classImage[:, 1:]
    boundary[:, 1:]  |= classImage[:, :-1] != classImage[:, 1:]
    boundary[:-1, :] |= classImage[:-1, :] != classImage[1:, :]
    boundary[1:, :]  |= classImage[:-1, :] != classImage[1:, :]
    return boundary

def computeBoundaryErrorImage(groundTruthImage, segmentationResultsImage,
                               boundaryLineWidth=1,
                               correctColor=(0.0, 0.45, 0.70),
                               falseNegativeColor=(0.90, 0.60, 0.0),
                               falsePositiveColor=(0.80, 0.0, 0.0)):
    """
    background/non boundary: white
    label and output boundaries agree: blue
    boundary in label but not output (false negative): orange
    boundary in output but not label (false positive): red
 
    boundaryLineWidth: pixel width to render each boundary line
    1 (default) does what it says on tin. 2 or 3 dilates the lines 
    outward purely for visuals, does not change underlying detection
    """
    labelBoundary = computeBoundaryMask(groundTruthImage)
    outputBoundary = computeBoundaryMask(segmentationResultsImage)
 
    if boundaryLineWidth > 1:
        structure = generate_binary_structure(2, 1)
        iterations = boundaryLineWidth - 1
        labelBoundary = binary_dilation(labelBoundary, structure=structure, iterations=iterations)
        outputBoundary = binary_dilation(outputBoundary, structure=structure, iterations=iterations)
 
    h, w = groundTruthImage.shape
    boundaryError = np.ones((h, w, 3), dtype=float)  # white background
 
    correct = labelBoundary & outputBoundary
    falseNegative = labelBoundary & ~outputBoundary
    falsePositive = ~labelBoundary & outputBoundary
 
    # correct first, then false negatives, then false positives on
    # top so that with dilation enabled errors never hide corrects
    boundaryError[correct] = correctColor
    boundaryError[falseNegative] = falseNegativeColor
    boundaryError[falsePositive] = falsePositiveColor
 
    return boundaryError

def _saveSubfigure(imageArray, savePath, cmap=None, vmin=None, vmax=None, dpi=300,
                    tickFontSize=14, tickLength=2, tickWidth=0.5,
                    borderWidth=0.6, paddingFraction=0.05):
    """
    Saves single tiles to be stitched later
    """
    h, w = imageArray.shape[:2]
    fig = plt.figure(figsize=(4, 4 * h / w))
    # leave room on all sides for padding + ticks/labels by insetting the
    # axes rather than using the previous full-bleed [0,0,1,1]
    pad = paddingFraction
    ax = fig.add_axes([pad, pad, 1 - 2 * pad, 1 - 2 * pad])
    ax.imshow(imageArray, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none",
          extent=(0, w, h, 0))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
 
    # xTicks = [0, w / 2, w]
    # yTicks = [0, h / 2, h]
    # ax.set_xticks(xTicks)
    # ax.set_xticklabels([f"{int(round(t))}" for t in xTicks], fontsize=tickFontSize)
    # ax.set_yticks(yTicks)
    # ax.set_yticklabels([f"{int(round(t))}" for t in yTicks], fontsize=tickFontSize)
    # ax.tick_params(length=tickLength, width=tickWidth, pad=1)

    ax.set_xticks([])
    ax.set_yticks([])

    # thin black border, useful for the boundary images or void in pastis ground truth
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(borderWidth)
        spine.set_color("black")
 
    fig.savefig(savePath, dpi=dpi, facecolor="white")
    plt.close(fig)

def _saveLegend(classIndices, classCmap, classNames, savePath, dpi=300,
                 borderWidth=0.8, paddingFraction=0.12, fontSize=12):
    """
    saves a tile's legend only as an image
    """
    fig = plt.figure(figsize=(2.2, 2.2))
    pad = paddingFraction
    ax = fig.add_axes([pad, pad, 1 - 2 * pad, 1 - 2 * pad])
 
    patches = [
        mpatches.Patch(facecolor=classCmap.colors[i], label=classNames[i], edgecolor="black")
        for i in sorted(classIndices)
    ]
    ax.legend(handles=patches, loc="center", frameon=False, fontsize=fontSize, title="Present Classes", title_fontsize=fontSize+4)
 
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        spine.set_linewidth(borderWidth)
        spine.set_color("black")
 
    fig.savefig(savePath, dpi=dpi, facecolor="white")
    plt.close(fig)

def viewBoundaryError(groundTruthImage, segmentationResultsImage, showLabels=True,
                       savePredsPath=None, saveBoundaryErrPath=None, saveLabelsPath=None,
                       saveLegendPath=None, boundaryLineWidth=1, saveDpi=300, skipShow=False,
                       correctColor=(0.0, 0.45, 0.70),
                       falseNegativeColor=(0.90, 0.60, 0.0),
                       falsePositiveColor=(0.80, 0.0, 0.0)):
    """
    Give outputs and labels, highlights class boundary errors
    """
    boundaryErrorImage = computeBoundaryErrorImage(
        groundTruthImage, segmentationResultsImage,
        boundaryLineWidth=boundaryLineWidth,
        correctColor=correctColor, falseNegativeColor=falseNegativeColor, falsePositiveColor=falsePositiveColor,
    )
 
    classVmax = len(cmapPastis.colors) - 1 # PASTIS
    # classVmax = len(cmapMTLCC.colors) - 1 # MTLCC
 
    if saveLabelsPath is not None:
        _saveSubfigure(groundTruthImage, saveLabelsPath, cmap=cmapPastis, vmin=0, vmax=classVmax, dpi=saveDpi) # manual change cmap
    if savePredsPath is not None:
        _saveSubfigure(segmentationResultsImage, savePredsPath, cmap=cmapPastis, vmin=0, vmax=classVmax, dpi=saveDpi) # manual change cmap
    if saveBoundaryErrPath is not None:
        _saveSubfigure(boundaryErrorImage, saveBoundaryErrPath, dpi=saveDpi)
    if saveLegendPath is not None:
        presentClasses = np.unique(groundTruthImage).tolist()
        _saveLegend(presentClasses, cmapPastis, pastisClasses, saveLegendPath, dpi=saveDpi) # manual change cmap and class list
 
    if showLabels:
        fig = plt.figure(figsize=(14, 4))
        grid = GridSpec(1, 4, width_ratios=[4, 4, 4, 2])
        axLabelImg = fig.add_subplot(grid[0])
        axOutputImg = fig.add_subplot(grid[1])
        axBoundaryImg = fig.add_subplot(grid[2])
        axLeg = fig.add_subplot(grid[3])
 
        axLabelImg.imshow(groundTruthImage, cmap=cmapPastis, vmin=0, vmax=classVmax, interpolation="none")
        axLabelImg.set_title("Labels")
        axLabelImg.axis("off")
    else:
        fig = plt.figure(figsize=(10, 4))
        grid = GridSpec(1, 3, width_ratios=[4, 4, 2])
        axOutputImg = fig.add_subplot(grid[0])
        axBoundaryImg = fig.add_subplot(grid[1])
        axLeg = fig.add_subplot(grid[2])
 
    axOutputImg.imshow(segmentationResultsImage, cmap=cmapPastis, vmin=0, vmax=classVmax, interpolation="none")
    axOutputImg.set_title("Outputs")
    axOutputImg.axis("off")
 
    axBoundaryImg.imshow(boundaryErrorImage, interpolation="none")
    axBoundaryImg.set_title("Boundary error")
    axBoundaryImg.axis("off")
 
    # class legend
    classPatches = [mpatches.Patch(facecolor=pastisColours[i], label=pastisClasses[i].format(l=pastisClasses[i]), edgecolor="Black") for i in range(len(pastisClasses))]
    classLegend = axLeg.legend(handles=classPatches, bbox_to_anchor=(-0.2, 1), loc=2, borderaxespad=0., fontsize=8, title="Classes")
    axLeg.add_artist(classLegend)
 
    # boundary-error legend
    boundaryPatches = [
        mpatches.Patch(facecolor=correctColor, label="Correct boundary", edgecolor="Black"),
        mpatches.Patch(facecolor=falseNegativeColor, label="False negative boundary", edgecolor="Black"),
        mpatches.Patch(facecolor=falsePositiveColor, label="False positive boundary", edgecolor="Black"),
    ]
    axLeg.legend(handles=boundaryPatches, bbox_to_anchor=(-0.2, 0.35), loc=2, borderaxespad=0., fontsize=8, title="Boundary error")
    axLeg.axis("off")
 
    if not skipShow:
        plt.show()