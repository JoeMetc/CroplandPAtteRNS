#####
# Author: Joseph Metcalfe
# Creation Date: 16/03/2026
# Purpose: Visualising PASTIS .npy files
#####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Patch
from datetime import date, datetime, timedelta
from os import path

##################################################

INPUTTILEDIR = "D:/Data/MTLCC/data_IJGI18/datasets/fullNpy/240/data16/"
INPUTTILEID = "7878"
STARTTEMPORALBAND = 0
ENDTEMPORALBAND = 1

# create a colour map to colour match identical crops in the one used by https://github.com/VSainteuf/pastis-benchmark/blob/main/documentation/pastis-documentation.pdf 
# non overlapping crops should get a colour not in the PASTIS colours
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
cmpMTLCC = ListedColormap(mtlccColours)

CLASS_NAMES_ORIGINAL = {
    0: "Unknown",
    1: "Sugar Beet",
    2: "Summer Oat",
    3: "Meadow",
    4: "Rape",
    5: "Hop",
    6: "Winter Spelt",
    7: "Winter Triticale",
    8: "Beans",
    9: "Peas",
    10: "Potatoe",
    11: "Soybeans",
    12: "Asparagus",
    13: "Winter Wheat",
    14: "Winter Barley",
    15: "Winter Rye",
    16: "Summer Barley",
    17: "Maize"
}

CLASS_NAMES = {
    0: "Background",
    1: "Sugar Beet",
    2: "Summer Oat",
    3: "Meadow",
    4: "Rapeseed",
    5: "Hop",
    6: "Winter Spelt",
    7: "Winter Triticale",
    8: "Beans",
    9: "Peas",
    10: "Potatoes",
    11: "Soybeans",
    12: "Asparagus",
    13: "Winter Wheat",
    14: "Winter Barley",
    15: "Winter Rye",
    16: "Summer Barley",
    17: "Maize"
}

def create_full_legend():
    legend_handles = []

    for class_id in range(len(mtlccColours)):
        legend_handles.append(
            Patch(
                facecolor=mtlccColours[class_id],
                edgecolor='black',
                label=f"{class_id}: {CLASS_NAMES[class_id]}"
            )
        )

    return legend_handles

def main():
    """The general function that allows selection and viewing of band data"""

    # read in the image file
    inputFileData = path.join(INPUTTILEDIR, f"{INPUTTILEID}_sits.npy")
    tileArrayData = np.load(inputFileData)
    inputFileAnnotation = path.join(INPUTTILEDIR, f"{INPUTTILEID}_labels.npy")
    tileArrayAnnotation = np.load(inputFileAnnotation)
    inputFileDates = path.join(INPUTTILEDIR, f"{INPUTTILEID}_dates.npy")
    tileArrayDates = np.load(inputFileDates)
    print(tileArrayDates)
    

    print(f"Data Shape: {np.shape(tileArrayData)}")
    print(f"Annotation Shape: {np.shape(tileArrayAnnotation)}")

    # Utility for quick check of which classes are present in a tile
    print(f"Unique annotation values: \n{np.unique(tileArrayAnnotation)}")

    # figLegend, axLegend = plt.subplots(figsize=(6, 8))

    # legend = axLegend.legend(
    #     handles=create_full_legend(),
    #     loc='center',
    #     frameon=False,
    #     markerfirst=True
    # )

    # # for text in legend.get_texts():
    # #     text.set_ha('right')

    # axLegend.axis('off')

    # # display the plots
    # plt.show()

    # # clear axes and figure
    # plt.close()

    # show every date in range
    for day in range(STARTTEMPORALBAND,ENDTEMPORALBAND):

        # create the subplot grid
        fig, axs = plt.subplots(ncols = 2)
        fig.suptitle("Crop Type Annotation vs Single Band Data for Single Temporal Band in MTLCC")

        axs[0].imshow(tileArrayAnnotation)
        axs[0].set_title("Labels")
        axs[0].axis('off')

        axs[1].imshow(tileArrayData[day][0], cmap = cmpMTLCC, interpolation="nearest")
        axs[1].set_title("X Channel")
        axs[1].axis('off')

        # display the plots
        plt.show()

        # clear axes and figure
        plt.close()

    DOY = False
    tilesDates = tileArrayDates
    if DOY: # if DOY just use the DOY value in the pairs
        tilesDates = tilesDates[:,1]
    else:
        tilesDates = [date(int(y), 1, 1) + timedelta(days=int(d) - 1) for y, d in tilesDates]

    print(tilesDates)
    

##################################################

if __name__ == "__main__":
    main()