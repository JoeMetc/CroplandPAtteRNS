#####
# Author: Joseph Metcalfe
# Creation Date: 30/04/2025
# Purpose: in-training model utility functions
#####

from os import path, remove
    
class EarlyStopping:
    """
    Class for early stopping based on patience for epochs without validation loss improvement.
    Implemented as class rather than function for persistent memory away from train val loop code.
    """
    def __init__(self, patience=1, allowedMargin=0, minEpochs=0):
        self.patience = patience
        self.allowedMargin = allowedMargin
        self.currentBestMIoU = float('-inf')
        self.badEpochs = 0
        self.minEpochs = minEpochs

    def earlyStopCheck(self, mIoU, currentEpoch):
        print(f"Epoch {currentEpoch}, bad epoch count = {self.badEpochs}")

        improved = mIoU > self.currentBestMIoU

        if improved:
            self.currentBestMIoU = mIoU
            self.badEpochs = 0
        else:
            if mIoU < self.currentBestMIoU - self.allowedMargin:
                self.badEpochs += 1

        if currentEpoch < self.minEpochs:
            return False

        return self.badEpochs >= self.patience
    
def pruneCheckpoints(rankedCheckpoints, label, keepCount):
    """
    Prune lowest-performing model checkpoints from a fold directory.
    rankedCheckpoints: list of (pthPath, mIoU) tuples, unsorted.
    retains the worst as a floor, plus the top N by mIoU.
    running multiple times on the same directory should be safe/wont whittle to 0 files remaining
    """
    if len(rankedCheckpoints) <= keepCount:
        print(f"[{label}] Too few checkpoints to prune, skipping.")
        return

    sorted_checkpoints = sorted(rankedCheckpoints, key=lambda x: x[1])  # ascending, worst first

    worst = sorted_checkpoints[0]
    remainder = sorted_checkpoints[1:]  # exclude worst from keep count

    toKeep = set(p for p, _ in remainder[-keepCount:])  # top keepCount from remainder
    toKeep.add(worst[0])  # always keep worst

    toDelete = [(p, m) for p, m in sorted_checkpoints if p not in toKeep]

    for pthPath, miou in toDelete:
        try:
            remove(pthPath)
            print(f"[{label}] Pruned {path.basename(pthPath)} (mIoU: {miou})")
        except OSError as e:
            print(f"[{label}] Failed to remove {path.basename(pthPath)}: {e}")

    print(f"[{label}] Pruning complete. Removed {len(toDelete)}/{len(sorted_checkpoints)} checkpoints, retained {len(toKeep)}.")