import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import cptio
import xarray as xr
import pandas as pd

#select file for specific lead time and start date
def sel_cpt_file(data_path, file_pattern, leadTime, startDate):
    pattern = f"{startDate}_{leadTime}"
    fullPath = f"{data_path}/{file_pattern}"
    fileName = fullPath.replace("*",pattern)
    fileSelected = cptio.open_cptdataset(fileName)
    startDT = datetime.strptime(startDate, "%b-%d-%Y")
    fileSelected = fileSelected.expand_dims({"S":[startDT]})
    return fileSelected


def target_range_format(leads_value,leads_key,start_date,period_length):
    ''' Formatting target range using leads and starts, and target range period length.
    '''
    target_start = start_date + timedelta(days=leads_value)
    target_end = target_start + timedelta(days= period_length - 1)
    if (target_start).strftime("%Y") == (target_end).strftime("%Y"):
        if (target_start).strftime("%b") == (target_end).strftime("%b"):
            target_start_str = target_start.strftime("%-d")
        else:
            target_start_str = (target_start).strftime("%-d %b")
    else:
        target_start_str = (target_start).strftime("%-d %b %Y")
    target_end_str = target_end.strftime("%-d %b %Y")
    date_range = f"{target_start_str} - {target_end_str}"
    return date_range


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
