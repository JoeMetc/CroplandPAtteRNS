#####
# Author: Joseph Metcalfe
# Purpose: Wrapper for safe config file IO
#####

from yaml import safe_load, dump

def yamlRead(file):
    with open(file, 'r') as configFile:
        inputDict = safe_load(configFile)
    return inputDict