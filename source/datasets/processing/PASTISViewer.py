#####
# Author: Joseph Metcalfe
# Creation Date: 20/11/2024
# Purpose: Visualising PASTIS .npy files
#####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from os import path

##################################################

# INPUTTILEDIR = "D:/Data/PASTIS/PASTISFresh/"
# INPUTTILEID = "40236"
INPUTTILEDIR = "D:/Data/PASTIS24/PASTISFresh/"
INPUTTILEID = "30394_16"
STARTTEMPORALBAND = 0
ENDTEMPORALBAND = 1

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
cmpPastis = ListedColormap(pastisColours)

CLASS_NAMES_ORIGINAL = {
    0: "Background",
    1: "Meadow",
    2: "Soft Winter Wheat",
    3: "Corn",
    4: "Winter Barley",
    5: "Winter Rapeseed",
    6: "Spring Barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter Triticale",
    11: "Winter Durum Wheat",
    12: "Fruits, Vegetables, & Flowers",
    13: "Potatoes",
    14: "Leguminous Fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed Cereal",
    18: "Sorghum",
    19: "Void"
}

CLASS_NAMES = {
    0: "Background",
    1: "Meadow",
    2: "Winter Soft Wheat",
    3: "Maize",
    4: "Winter Barley",
    5: "Winter Rapeseed",
    6: "Spring Barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Sugar Beet",
    10: "Winter Triticale",
    11: "Winter Durum Wheat",
    12: "Fruits, Vegetables, & Flowers",
    13: "Potatoes",
    14: "Leguminous Fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed Cereal",
    18: "Sorghum",
    19: "Void"
}

def create_full_legend():
    legend_handles = []

    for class_id in range(len(pastisColours)):
        legend_handles.append(
            Patch(
                facecolor=pastisColours[class_id],
                edgecolor='black',
                label=f"{class_id}: {CLASS_NAMES[class_id]}"
            )
        )

    return legend_handles



def main():
    """The general function that allows selection and viewing of band data"""

    # read in the image file
    inputFileData = path.join(INPUTTILEDIR, "DATA_S2/S2_" + str(INPUTTILEID) + ".npy")
    tileArrayData = np.load(inputFileData)
    print(np.shape(tileArrayData))
        # TARGET_ seems to be the class in layer 0, a seemingly random count/order of parcels within the tile in layer 1, and something still undetermined in layer 3
    inputFileAnnotation = path.join(INPUTTILEDIR, "ANNOTATIONS/TARGET_" + str(INPUTTILEID) + ".npy")
        # ParcelIDs_ contains the unique ID for each parcel, although no directory of these is known, not used by us
    # inputFileAnnotation = path.join(INPUTTILEDIR, "ANNOTATIONS/ParcelIDs_" + str(INPUTTILEID) + ".npy")
        # HEATMAP_ contains the 'centerness heatmap' of each tile, not used by us
    # inputFileAnnotation = path.join(INPUTTILEDIR, "INSTANCE_ANNOTATIONS/HEATMAP_" + str(INPUTTILEID) + ".npy")
        # ZONES contains a single layer map of i think which pixel belongs to which nearest centre?
    # inputFileAnnotation = path.join(INPUTTILEDIR, "INSTANCE_ANNOTATIONS/ZONES_" + str(INPUTTILEID) + ".npy")
    tileArrayAnnotation = np.load(inputFileAnnotation)

    # print(f"Data Shape: {np.shape(tileArrayData)}")
    # print(f"Annotation Shape: {np.shape(tileArrayAnnotation)}")

    # Utility for quick check of which classes are present in a tile
    print(f"Unique annotation values: \n{np.unique(tileArrayAnnotation)}")

    #########
    # use this to just get a full legend
    # figLegend, axLegend = plt.subplots(figsize=(6, 8))

    # legend = axLegend.legend(
    #     handles=create_full_legend(),
    #     loc='center',
    #     frameon=False,
    #     markerfirst=False
    # )

    # for text in legend.get_texts():
    #     text.set_ha('left')

    # axLegend.axis('off')

    # plt.show()
    # plt.close()
    #########

    # show every date in range
    for date in range(STARTTEMPORALBAND,ENDTEMPORALBAND):

        # create the subplot grid
        fig, axs = plt.subplots(ncols = 2)
        fig.suptitle("Crop Type Annotation vs Single Band Data for Single Temporal Band in PASTIS")

        axs[0].imshow(tileArrayAnnotation, cmap = cmpPastis, interpolation="none", vmin = 0, vmax = 19)
        axs[0].set_title("Labels")
        axs[0].axis('off')

        axs[1].imshow(tileArrayData[date][0], cmap = "gray", interpolation="none")
        axs[1].set_title("X Channel")
        axs[1].axis('off')

        # display the plots
        plt.show()

        # clear axes and figure
        plt.close()

##################################################

if __name__ == "__main__":
    main()