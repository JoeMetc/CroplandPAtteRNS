#####
# Author: Joseph Metcalfe
# Creation Date: 16/03/2025
# Purpose: Specifc data loader for MTLCC 48 data held locally
#####

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, default_collate
import numpy as np
import pandas as pd
# import geopandas as gpd
from datetime import date, datetime, timedelta
from os import path
        
def utae_collate_fn(batch):
    idxs, sits_list, dates_list, labels = zip(*batch)

    # Convert everything to tensors
    idxs = [torch.as_tensor(i, dtype=torch.int32) for i in idxs]
    sits_tensors = [torch.as_tensor(s, dtype=torch.float32) for s in sits_list]
    dates_tensors = [torch.as_tensor(d, dtype=torch.float32) for d in dates_list]
    labels_tensors = [torch.as_tensor(l, dtype=torch.long) for l in labels]

    # Pad temporal dimension (T) so all sequences in batch have equal length
    sits_padded = pad_sequence(sits_tensors, batch_first=True, padding_value=0)  # [B, T_max, 10, 128, 128]
    dates_padded = pad_sequence(dates_tensors, batch_first=True, padding_value=0)  # [B, T_max]
    labels_stacked = torch.stack(labels_tensors)  # [B, 128, 128]

    return idxs, sits_padded, dates_padded, labels_stacked

def MTLCC48Dataloader(config, rootDir, samplesFilePath, maskBackground, batchSize, numWorkers, shuffle, seed, transform=None):
    # custom shaped dataset for multispectral multitemporal sat images
    dataset = S2Data(config, rootDir, samplesFilePath, maskBackground, transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if config["MODEL"]["dates"] == "paired":
        dataloader = DataLoader(dataset, batch_size=batchSize, collate_fn=utae_collate_fn, shuffle=shuffle, num_workers=numWorkers, pin_memory=True, persistent_workers=True, generator=generator)
    else:
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers, pin_memory=True, persistent_workers=True, generator=generator)

    return dataloader

def get_doy(date): # fixed
    date = str(date)
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.strptime(date, '%Y.%m.%d')
    return float(dt.timetuple().tm_yday)


class S2Data(Dataset):
    """Class for Sentinel 2 Multispectral Multitemporal imagery with .npy data and annotation files"""

    def __init__(self, config, rootDir, samplesFilePath, maskBackground, transform=None):
        
        # check root exists
        if not path.isdir(rootDir):
            raise Exception("Root Directory Not Found")
        
        self.config = config
        self.rootDir = rootDir
        self.tilePaths = pd.read_csv(path.join(rootDir, samplesFilePath))
        self.transform = transform
        self.startDate = date.fromisoformat(str(20160102)) # One day before first date (avoid date encoding of 0), these are for the 2016 version
        # self.startDate = date.fromisoformat(str(20160103)) # Archive!!  # These are for the 2016 version, the 2017 has different labels and is much shorter so not currently using
        self.endDate = date.fromisoformat(str(20161208)) # length of 342 days inclusive, 2016 was a leap year
        self.maskBackground = maskBackground
        self.temporalEmbeddingLength = self.config["MODEL"]["tempEmbeddingLength"]

        self.tileDatesDict = {}

        self.DOY = False # False = Days Since Dataset Started, True = Day of Year for loader date encoding
        if self.config["MODEL"]["dateOrder"] == "DOY":
            self.DOY = True
        
    def __len__(self):
        return len(self.tilePaths)

    def __getitem__(self, dataIndex):
        imgPath = self.tilePaths.loc[dataIndex, "imagepath"]
        labelPath = self.tilePaths.loc[dataIndex, "labelpath"]
        datePath = self.tilePaths.loc[dataIndex, "datepath"]
        tileID = int((imgPath.split("_")[0].split("/")[-1]))

        sits = torch.from_numpy(np.load(path.join(self.rootDir, imgPath)).astype(np.float32)) # [0] for 3D (spectral), [0][0] for 2D - put into tensor later after temporal padding
        label = torch.from_numpy((np.load(path.join(self.rootDir, labelPath))).astype(np.int64)) # class labels on channel 0 of PASTIS annotation file
                                
        if self.transform:
            sits = self.transform(sits)

        # normalise sits
        sits.clamp_(0, 10000).div_(10000) # S2 digital data range is 0-10000

        tilesDates = np.load(path.join(self.rootDir, datePath)) # [Year, DOY]
        if self.DOY: # if DOY just use the DOY value in the pairs
            tileDates = tilesDates[:,1]
        else:
            tileDates = [date(int(y), 1, 1) + timedelta(days=int(d) - 1) for y, d in tilesDates]

        if self.config["MODEL"]["dates"] == "spectralChannel": # PAtteRNS, TSViT

            # earliest date (-1) in PASTIS by manual search is 20180916, latest is 20191027, giving 407 inclusive days of length
            if self.DOY:
                captureDays = torch.tensor(tileDates, dtype=torch.float32) / 365.0
            else:
                captureDays = torch.tensor([(d - self.startDate).days for d in tileDates],dtype=torch.float32) / self.temporalEmbeddingLength

            captureDays = captureDays.view(sits.shape[0], 1, 1, 1)
            temporalTokenChannel = captureDays.expand(-1, 1, 48, 48)
            tokenisedTemporalBands = torch.cat((temporalTokenChannel, sits), dim=1)

            # pad out the temporal lengths to the max length in the dataset
            temporalTokenSITS = torch.zeros((45, 14, 48, 48), dtype=sits.dtype)
            temporalTokenSITS[:len(tokenisedTemporalBands)] = tokenisedTemporalBands

            # Returning data index also to support pre-computed matrices
            return dataIndex, temporalTokenSITS, label
        
        elif self.config["MODEL"]["dates"] == "paired": # UTAE
            # use ref date provided by utae paper (changed to near mtlcc first observation date)
            refDate = date(*map(int, "2016-01-01".split("-")))
            daysSinceRefDateList = [(imgDate - refDate).days for imgDate in tileDates]

            # pad out the temporal lengths to the max length in the dataset
            paddedSITS = torch.zeros((45, 13, 48, 48), dtype=sits.dtype)
            paddedDates = torch.zeros(45, dtype=torch.float32)

            paddedSITS[:len(sits)] = sits
            paddedDates[:len(daysSinceRefDateList)] = torch.tensor(daysSinceRefDateList, dtype=torch.float32)

            return dataIndex, sits, daysSinceRefDateList, label
