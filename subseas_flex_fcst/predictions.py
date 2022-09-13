import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import cptio
import xarray as xr
import pandas as pd

#select file for specific lead time and start date
def selFile(dataPath, filePattern, leadTime, startDate):
    pattern = f"{startDate}_{leadTime}"
    fullPath = f"{dataPath}/{filePattern}"
    fileName = fullPath.replace("*",pattern)
    fileSelected = cptio.open_cptdataset(fileName)
    startDT = datetime.strptime(startDate, "%b-%d-%Y")
    fileSelected = fileSelected.expand_dims({"S":[startDT]})
    return fileSelected

# slightly outdates function to open all datasets and then combine them
def combine_cptdataset(dataDir,filePattern,dof=False):
    filesNameList = glob.glob(f'{dataDir}/{filePattern}')
    dataDic = {}
    for idx, i in enumerate(filesNameList):
        fileCompList = re.split('[_.]',filesNameList[idx])
        fileCompList = [x for x in fileCompList if "/" not in x]

        leadTimeWk = [x for x in fileCompList if x.startswith('wk')]

        dateFormat = re.compile(".*-.*-.*")
        startDateStr = list(filter(dateFormat.match,fileCompList))
        startDate = startDateStr[0]
        if leadTimeWk[0] == 'wk1':
            leadTimeDate = 'Week 1'
        elif leadTimeWk[0] == 'wk2':
            leadTimeDate = 'Week 2'
        elif leadTimeWk[0] == 'wk3':
            leadTimeDate = 'Week 3'
        elif leadTimeWk[0] == 'wk4':
            leadTimeDate = 'Week 4'
        leadTimeDict = {'L':[leadTimeDate]}

        ds = cptio.open_cptdataset(filesNameList[idx])
        if dof == True:
            dofVar = float(ds["tp_pred_err_var"].attrs["dof"])
            ds = ds.assign(dof=dofVar)
        if len(ds['T']) == 1:
            ds['T'] = [startDate] #for obs data there are many dates
        ds = ds.expand_dims(leadTimeDict)
        dataDic[idx] = ds
    dataCombine = xr.combine_by_coords([dataDic[x] for x in dataDic],combine_attrs="drop_conflicts")
    return dataCombine
