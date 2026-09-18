#####
# Original Source: https://github.com/michaeltrs/DeepSatModels/blob/main/data/PASTIS24/data2windows.py
# Adaptation: by Joseph Metcalfe to remove bugs where needed and adapt to .npy file output
# Purpose: Splitting PASTIS-128 tiles out into PASTIS-24 tiles
#####

import geopandas as gpd
import os
import numpy as np
import datetime
import torch
import argparse


def get_doy(date):
    date = str(date)
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.datetime.strptime(date, '%Y.%m.%d')
    return dt.timetuple().tm_yday


def unfold_reshape(img, HW):
    if len(img.shape) == 4:
        T, C, H, W = img.shape
        img = img.unfold(2, size=HW, step=HW).unfold(3, size=HW, step=HW)
        img = img.reshape(T, C, -1, HW, HW).permute(2, 0, 1, 3, 4)
    elif len(img.shape) == 3:
        _, H, W = img.shape
        img = img.unfold(1, size=HW, step=HW).unfold(2, size=HW, step=HW)
        img = img.reshape(-1, HW, HW)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PASTIS-24 tile writer")
    parser.add_argument('--rootdir', type=str, default="C:/Your/Filepath/Here/PASTIS/PASTISFresh",
                        help='PASTIS24 root dir')
    parser.add_argument('--savedir', type=str, default="C:/Your/Filepath/Here/PASTIS24/PASTISFresh",
                        help='where to save new data')
    parser.add_argument('--HWout', type=int, default=24,
                        help='size of extracted windows')
    args = parser.parse_args()

    rootdir = args.rootdir
    savedir = args.savedir
    HWin = 128
    HWout = args.HWout

    meta_patch = gpd.read_file(os.path.join(rootdir, "metadata.geojson"))

    img_save_dir = os.path.join(savedir, "DATA_S2")
    lab_save_dir = os.path.join(savedir, "ANNOTATIONS")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(lab_save_dir, exist_ok=True)

    labels = []
    for i in range(meta_patch.shape[0]):
        print('doing file %d of %d' % (i, meta_patch.shape[0]))
        img = np.load(os.path.join(rootdir, 'DATA_S2/S2_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        lab = np.load(os.path.join(rootdir, 'ANNOTATIONS/TARGET_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        ids = np.load(os.path.join(rootdir, 'ANNOTATIONS/ParcelIDs_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        dates = meta_patch['dates-S2'].iloc[i]
        doy = np.array([get_doy(d) for d in dates.values()])
        idx = np.argsort(doy) # DOY not used by the save functions as this messes up the order
        img = img[idx]
        doy = doy[idx]
        unfolded_images = unfold_reshape(torch.tensor(img), HWout).numpy()
        unfolded_labels = unfold_reshape(torch.tensor(lab), HWout).numpy()

        for tile_idx in range(unfolded_images.shape[0]):
            img_tile_path = os.path.join(img_save_dir, f"S2_{meta_patch['ID_PATCH'].iloc[i]}_{tile_idx}.npy")
            lab_tile_path = os.path.join(lab_save_dir, f"TARGET_{meta_patch['ID_PATCH'].iloc[i]}_{tile_idx}.npy")
            np.save(img_tile_path, unfolded_images[tile_idx])
            np.save(lab_tile_path, unfolded_labels[tile_idx])