#####
# Author: Joseph Metcalfe
# Creation Date: 16/11/2025
# Purpose: Specifc data loader for recreated PASTIS 24 data held locally
#####

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, default_collate
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import date, datetime
from os import path
import json
        
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

def PASTIS24Dataloader(config, rootDir, samplesFilePath, metadataPath, maskBackground, batchSize, numWorkers, shuffle, seed, transform=None):
    # custom shaped dataset for multispectral multitemporal sat images
    dataset = S2Data(config, rootDir, samplesFilePath, metadataPath, maskBackground, transform)

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

    def __init__(self, config, rootDir, samplesFilePath, metadataPath, maskBackground, transform=None):
        
        # check root exists
        if not path.isdir(rootDir):
            raise Exception("Root Directory Not Found")
        
        self.config = config
        self.rootDir = rootDir
        self.tilePaths = pd.read_csv(path.join(rootDir, samplesFilePath))
        self.transform = transform
        self.metadataPath = metadataPath
        self.startDate = date.fromisoformat(str(20180916)) # set to 1 day less than 20180917 to avoid dates encoding as padding value of 0
        # self.startDate = date.fromisoformat(str(20180917)) # Archive !!!!! # set to 1 day less than 20180917 to avoid dates encoding as padding value of 0
        self.endDate = date.fromisoformat(str(20191027))
        self.maskBackground = maskBackground

        self.tilesMetadata = gpd.read_file(path.join(self.rootDir, self.metadataPath))
        self.tilesMetadata["id"] = self.tilesMetadata["id"].astype(str)
        self.tilesMetadata.set_index("id", inplace=True)
        self.temporalEmbeddingLength = self.config["MODEL"]["tempEmbeddingLength"]

        self.tileDatesDict = {}

        self.DOY = False # False = Days Since Dataset Started, True = Day of Year for loader date encoding
        if self.config["MODEL"]["dateOrder"] == "DOY":
            self.DOY = True

        for tile_id, row in self.tilesMetadata.iterrows():
            try:
                dates = json.loads(row["dates-S2"]).values()
            except:
                dates = dict(row["dates-S2"]).values()

            if self.DOY:
                self.tileDatesDict[int(tile_id)] = list(dates)
            else:
                self.tileDatesDict[int(tile_id)] = [date.fromisoformat(str(d)) for d in dates]
        
    def __len__(self):
        return len(self.tilePaths)

    def __getitem__(self, dataIndex):
        imgPath = self.tilePaths.loc[dataIndex, "imagepath"]
        labelPath = self.tilePaths.loc[dataIndex, "labelpath"]
        imgPath = imgPath.replace("\\", "/")
        labelPath = labelPath.replace("\\", "/")
        tileID = self.tilePaths.loc[dataIndex, "fieldID"]
        parentID = self.tilePaths.loc[dataIndex, "parentID"]

        sits = torch.from_numpy(np.load(path.join(self.rootDir, imgPath)).astype(np.float32))
        label = torch.from_numpy((np.load(path.join(self.rootDir, labelPath))).astype(np.int64))

        if self.transform:
            sits = self.transform(sits)

        if self.maskBackground:
            # change the void labels (19) to background (0) for masked cross entropy as it can only mask one class index (so '0' represents both masked classes)
            label[label == 19] = 0

        # normalise sits
        sits.clamp_(0, 10000).div_(10000) # S2 digital data range is 0-10000

        tileDates = self.tileDatesDict[parentID]

        if self.config["MODEL"]["dates"] == "spectralChannel": # PAtteRNS, TSViT

            # earliest date (-1) in PASTIS by manual search is 20180916, latest is 20191027, giving 407 inclusive days of length
            if self.DOY:
                captureDays = torch.tensor([get_doy(d) for d in tileDates], dtype=torch.float32) / 365.0
            else:
                captureDays = torch.tensor([(d - self.startDate).days for d in tileDates],dtype=torch.float32) / self.temporalEmbeddingLength

            captureDays = captureDays.view(sits.shape[0], 1, 1, 1)
            temporalTokenChannel = captureDays.expand(-1, 1, 24, 24)
            tokenisedTemporalBands = torch.cat((temporalTokenChannel, sits), dim=1)

            # pad out the temporal lengths to the max length in the dataset
            temporalTokenSITS = torch.zeros((61, 11, 24, 24), dtype=sits.dtype)
            temporalTokenSITS[:len(tokenisedTemporalBands)] = tokenisedTemporalBands

            # Returning data index also to support pre-computed matrices
            return dataIndex, temporalTokenSITS, label
        
        elif self.config["MODEL"]["dates"] == "paired": # UTAE
            # use ref date provided by utae paper
            refDate = date(*map(int, "2018-09-01".split("-"))) # length of 422?
            daysSinceRefDateList = [(imgDate - refDate).days for imgDate in tileDates]

            # pad out the temporal lengths to the max length in the dataset
            paddedSITS = torch.zeros((61, 10, 24, 24), dtype=sits.dtype)
            paddedDates = torch.zeros(61, dtype=torch.float32)

            paddedSITS[:len(sits)] = sits
            paddedDates[:len(daysSinceRefDateList)] = torch.tensor(daysSinceRefDateList, dtype=torch.float32)

            return dataIndex, sits, daysSinceRefDateList, label
