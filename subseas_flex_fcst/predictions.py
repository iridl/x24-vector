import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import cptio
import xarray as xr
import pandas as pd

#select file for specific lead time and start date
def sel_cpt_file(data_path, filename_pattern, leadTime, startDate):
    """ Select a single cpt file for a given start and lead.

    Parameters
    ----------
    data_path : str
        String of the path pointing to cpt datasets.
    filename_pattern : str
        String of the filename pattern name for a given variable's data file.
    leadTime : str
         String of the lead time value to be selected for as is represented in the file name.
    startDate : str
        String of the start date to be selected for as is represented in the file name.
    Returns
    -------
    fileSelected : xarray Dataset
        Single CPT data file as multidimensional xarray dataset.
    Notes
    -----
    `filename_pattern` should be most common denominator for any group of datasets,
    so that a single file can be selected using only `leadTime` and `startDate`.
    Examples
    --------
    For files which have naming structure such as the example file: 
        CFSv2_SubXPRCP_CCAFCST_mu_Apr_Apr-1-2022_wk1.txt
    And where this file's `leadTime` and `startDate`:
        `leadTime` == 'wk1' and `startDate` == 'Apr-1-2022'
    `filename_pattern` == 'CFSv2_SubXPRCP_CCAFCST_mu_Apr_*.txt'
    """
    pattern = f"{startDate}_{leadTime}"
    fullPath = f"{data_path}/{filename_pattern}"
    fileName = fullPath.replace("*",pattern)
    fileSelected = cptio.open_cptdataset(fileName)
    startDT = datetime.strptime(startDate, "%b-%d-%Y")
    fileSelected = fileSelected.expand_dims({"S":[startDT]})
    return fileSelected


def target_range_format(leads_value,leads_key,start_date,period_length):
    """ Formatting target range using leads and starts, and target range period length.

    Parameters
    ----------
    leads_value : int
        Application's integer representation of lead time.
    leads_key : str
        Provider's representation of lead time in the data.
    start_date : Timestamp
        Start date for the forecast.
    period_length : int
       Length of forecast target range period.
    Returns
    -------
    date_range : str
        String of target date range.
    Notes
    -----
    If the providers representation of lead time is an integer value, convert to str for input.
    The function will output the most concise version of the date range depending on if years and months are equal.
    """
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


def cpt_starts_list(data_path,filename_pattern,regex_search_pattern,strftime_format="%b-%-d-%Y"):
    """ Get list of all start dates from CPT files.

    Parameters
    ----------
    data_path : str
        String of the path pointing to cpt datasets.
    filename_pattern : str
        String of the filename pattern name for a given variable's data file.
    regex_search_pattern : str
        String representing regular expression search pattern to find dates in file names.
    strftime_format : str
        String representing output dates format.
    Returns
    -------
    start_dates : list
        List of strings representing all start dates for the data within `data_path`.
    Notes
    -----
    For more information on regex visit: https://docs.python.org/3/library/re.html
    Test your regex code here: https://regexr.com/
    Examples
    --------
    Regex expression "\w{3}-\w{1,2}-\w{4}" matches expressions that are:
    '{word of 3 chars}-{word between 1,2 chars}-{word of 4 chars}'
    will match dates of format 'Apr-4-2022', 'dec-14-2022', etc.
    """
    files_name_list = glob.glob(f'{data_path}/{filename_pattern}')
    start_dates = []
    for file in files_name_list:
        start_date = re.search(regex_search_pattern,file)
        start_date_dt = datetime.strptime(start_date.group(),"%b-%d-%Y")
        start_dates.append(start_date_dt)
    start_dates = sorted(set(start_dates)) #finds unique dates in the case there are files with the same date due to multiple lead times
    start_dates = [i.strftime(strftime_format) for i in start_dates]
    return start_dates
