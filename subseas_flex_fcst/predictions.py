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

#function to get target start, end using issue date and lead time
def getTargets(issueDate, leadTime, tp_length):
    # Get Issue date and Target season
    issue_date_td = pd.to_datetime(issueDate) #fcst_var["S"].values
    issue_date = issue_date_td[0].strftime("%-d %b %Y")
    #for leads they are currently set to be the difference in days fron target_start to issue date
    if leadTime == "wk1":
        lead_time = 1
    elif leadTime == "wk2":
        lead_time = 8
    elif leadTime == "wk3":
        lead_time = 15
    elif leadTime == "wk4":
        lead_time = 22
    target_start = (issue_date_td + timedelta(days=lead_time))[0].strftime("%-d %b %Y")
    target_end = (issue_date_td + timedelta(days=(lead_time+tp_length-1)))[0].strftime("%-d %b %Y")

    return issue_date, target_start, target_end

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


def cpt_startsList(dataPath,filePattern,dateSearchPattern,zeroPadding=False):
    filesNameList = glob.glob(f'{dataPath}/{filePattern}')
    startDates = []
    for file in filesNameList:
        startDate = re.search(dateSearchPattern,file)
        startDatedt = datetime.strptime(startDate.group(),"%b-%d-%Y")
        startDates.append(startDatedt)
    startDates = sorted(set(startDates))
    if zeroPadding == False:
        startDates = [i.strftime("%b-%-d-%Y") for i in startDates] #needs to have date with no zero padding to match the file path namesme
    if zeroPadding == True:
        startDates = [i.strftime("%b-%d-%Y") for i in startDates]
    return startDates
