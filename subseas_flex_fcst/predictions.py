import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import cptio
import xarray as xr
import pandas as pd

#select file for specific lead time and start date
def selFile(data_path, file_pattern, leadTime, startDate):
    pattern = f"{startDate}_{leadTime}"
    fullPath = f"{data_path}/{file_pattern}"
    fileName = fullPath.replace("*",pattern)
    fileSelected = cptio.open_cptdataset(fileName)
    startDT = datetime.strptime(startDate, "%b-%d-%Y")
    fileSelected = fileSelected.expand_dims({"S":[startDT]})
    return fileSelected

# slightly outdates function to open all datasets and then combine them
def combine_cptdataset(dataDir,file_pattern,dof=False):
    files_name_list = glob.glob(f'{dataDir}/{file_pattern}')
    dataDic = {}
    for idx, i in enumerate(files_name_list):
        fileCompList = re.split('[_.]',files_name_list[idx])
        fileCompList = [x for x in fileCompList if "/" not in x]

        leadTimeWk = [x for x in fileCompList if x.startswith('wk')]

        dateFormat = re.compile(".*-.*-.*")
        start_datestr = list(filter(dateFormat.match,fileCompList))
        start_date= start_datestr[0]
        if leadTimeWk[0] == 'wk1':
            leadTimeDate = 'Week 1'
        elif leadTimeWk[0] == 'wk2':
            leadTimeDate = 'Week 2'
        elif leadTimeWk[0] == 'wk3':
            leadTimeDate = 'Week 3'
        elif leadTimeWk[0] == 'wk4':
            leadTimeDate = 'Week 4'
        leadTimeDict = {'L':[leadTimeDate]}

        ds = cptio.open_cptdataset(files_name_list[idx])
        if dof == True:
            dofVar = float(ds["tp_pred_err_var"].attrs["dof"])
            ds = ds.assign(dof=dofVar)
        if len(ds['T']) == 1:
            ds['T'] = [startDate] #for obs data there are many dates
        ds = ds.expand_dims(leadTimeDict)
        dataDic[idx] = ds
    dataCombine = xr.combine_by_coords([dataDic[x] for x in dataDic],combine_attrs="drop_conflicts")
    return dataCombine

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

def cpt_starts_list(data_path,file_pattern,date_search_pattern,zero_padding=False):
    files_name_list = glob.glob(f'{data_path}/{file_pattern}')
    start_dates = []
    for file in files_name_list:
        start_date = re.search(date_search_pattern,file)
        start_date_dt = datetime.strptime(start_date.group(),"%b-%d-%Y")
        start_dates.append(start_date_dt)
    start_dates = sorted(set(start_dates))
    if zero_padding == False:
        start_dates = [i.strftime("%b-%-d-%Y") for i in start_dates] #needs to have date with no zero padding to match the file path namesme
    if zero_padding == True:
        start_dates = [i.strftime("%b-%d-%Y") for i in start_dates]
    return start_dates
