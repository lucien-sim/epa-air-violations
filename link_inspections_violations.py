#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

def load_raw_inspections(file_path,start_date=pd.Timestamp(year=2013,month=1,day=1),
                     end_date=pd.Timestamp(year=2018,month=12,day=31)):
    """Loads raw inspection data (directly from downloaded file), returns cleaned/parsed dataframe."""
    
    # Load data, parse dates manually (didn't work in load function for some reason?)
    fces_pces = pd.read_csv(file_path,dtype=str,parse_dates=['ACTUAL_END_DATE'])
    fces_pces['ACTUAL_END_DATE'] = fces_pces['ACTUAL_END_DATE'].apply(
                            lambda x: datetime.datetime.strptime(x,'%m-%d-%Y'))
    
    # Restrict dates to specified time period
    fces_pces = fces_pces[(fces_pces['ACTUAL_END_DATE']>=start_date) & 
                          (fces_pces['ACTUAL_END_DATE']<=end_date)]
    
    # Drop inspections without a date, sort. 
    fces_pces = fces_pces.dropna(axis=0,subset=['ACTUAL_END_DATE'])
    fces_pces = fces_pces.sort_values(by=['PGM_SYS_ID','ACTUAL_END_DATE'],axis=0)
    
    return fces_pces


def time_since_prev(df,date_col='ACTUAL_END_DATE',group_col='PGM_SYS_ID'): 
    """Adds new column: time since the previous inspection"""
    if group_col is not None: 
        time_diff = df.groupby(group_col)[date_col].diff()/pd.Timedelta('1 days')
        time_diff[time_diff.isnull()]=np.NaN
        time_diff = time_diff.astype(float)
    else: 
        time_diff = df[date_col].diff()/pd.Timedelta('1 days')
        time_diff[time_diff.isnull()]=np.NaN
        time_diff = time_diff.astype(float)
    return time_diff


def merge_events(df,time_since_col='time_since_prev',days_thresh=14): 
    """Merge inspection events that occur within 14 days of each other"""
    return df[((df[time_since_col]>days_thresh)) | (df[time_since_col].isna())]


def load_violations(file_path,start_date=pd.Timestamp(year=2003,month=1,day=1),
                     end_date=pd.Timestamp(year=2018,month=12,day=31)):
    """Loads inspection data directly from downloaded files, returns parsed and cleaned dataframe"""
    
    # Read in data
    violations = pd.read_csv(file_path,dtype='str')
    
    viol_dates = violations['EARLIEST_FRV_DETERM_DATE'].copy()
    viol_dates[violations['EARLIEST_FRV_DETERM_DATE'].isna()] = violations['HPV_DAYZERO_DATE'][
                                                            violations['EARLIEST_FRV_DETERM_DATE'].isna()]
    violations['VIOL_DATE'] = viol_dates
    violations = violations.dropna(axis=0,subset=['VIOL_DATE'])
    violations['VIOL_DATE'] = violations['VIOL_DATE'].apply(lambda x: 
                                                            datetime.datetime.strptime(x,'%m-%d-%Y'))
    
    # Restrict dates
    violations = violations[(violations['VIOL_DATE']>=start_date) & 
                            (violations['VIOL_DATE']<=end_date)]
    
    # Sort by SOURCE ID and date. 
    violations = violations.sort_values(by=['PGM_SYS_ID','VIOL_DATE'],axis=0)
    
    return violations


def link_viol_insp(inspections,violations,past_thresh=0,future_thresh=90):
    
    # Dictionary of violations--for easy access later. 
    viol_dict = {}
    for source_id,source_data in violations[['PGM_SYS_ID','VIOL_DATE','ACTIVITY_ID'
                                            ]].groupby('PGM_SYS_ID')[['ACTIVITY_ID','VIOL_DATE']]: 
        viol_dict[source_id] = source_data
    
    # Loop through all sources--link violations at that source to inspections. 
    pgmsysid,inspdate,violtf,actid = [],[],[],[]
    insp_dates_by_source = inspections[['PGM_SYS_ID','ACTUAL_END_DATE']].groupby('PGM_SYS_ID')['ACTUAL_END_DATE']
    for source_id,insp_dates in insp_dates_by_source:

        # Initialize lists/arrays to store for each inspection: 
        insp_dates = np.asarray(insp_dates) # inspection dates
        viol_tf = np.zeros(np.shape(insp_dates)) # inspection pass/fail
        act_ids = [np.NaN]*len(insp_dates) # violation activity ids (when applicable)
        viol_data = viol_dict.get(source_id)

        # If there has been a violation at the source, mark corresponding inspections as failed and 
        # record violation activity ID for the violation associated with the inspection. 
        if viol_data is not None: 
            viol_dates = list(viol_data['VIOL_DATE'])
            viol_actids = list(viol_data['ACTIVITY_ID'])
            for viol_date,viol_actid in zip(viol_dates,viol_actids): 
                time_diffs = (viol_date-insp_dates)/datetime.timedelta(days=1)
                time_diffs[(time_diffs<past_thresh) | (time_diffs>future_thresh)] = np.NaN
                try: 
                    ind_viol = np.nanargmin(time_diffs)
                    if viol_tf[ind_viol]==0:
                        viol_tf[ind_viol] = 1
                        act_ids[ind_viol] = str(viol_actid)
                except ValueError:
                    pass

        # Accumulate inspection results in long vectors. 
        pgmsysid.extend([source_id]*len(insp_dates))
        inspdate.extend(list(insp_dates))
        violtf.extend(list(viol_tf))
        actid.extend(list(act_ids))

    # Create dataframe with inspection results. Add results to inspections dataframe with merge. 
    for_join = pd.DataFrame({
        'PGM_SYS_ID':pgmsysid,
        'ACTUAL_END_DATE':inspdate,
        'VIOL':violtf,
        'VIOL_ACTID':actid
    })
    for_join['DATE_JOIN'] = for_join['ACTUAL_END_DATE'].apply(str)
    for_join = for_join.drop('ACTUAL_END_DATE',axis=1)
    inspections['DATE_JOIN'] = inspections['ACTUAL_END_DATE'].apply(str)
    new_df = pd.merge(inspections, for_join,  how='left', on=['PGM_SYS_ID','DATE_JOIN']).drop('DATE_JOIN',
                                                                                              axis=1)  
    # Adjust datatypes of new columns
    new_df['VIOL'] = new_df['VIOL'].astype(dtype=int)
    new_df['VIOL_ACTID'] = new_df['VIOL_ACTID'].astype(dtype=str)
    
    return new_df


if __name__=='__main__': 

    log = open('logfile.txt','a')

    # Set some constants. 
    from external_variables import data_path
    start_date = pd.Timestamp(year=2003,month=1,day=1)
    end_date = pd.Timestamp(datetime.datetime.today())
    insp_viol_thresh_days = 365
    insp_event_thresh = 14

    # Step 1: Load inspection data
    inspections_path = os.path.join(data_path,'ICIS-Air','ICIS-AIR_FCES_PCES.csv')
    inspections = load_raw_inspections(inspections_path,start_date=start_date,end_date=end_date)

    # Step 2: Clean inspection data
    inspections['time_since_prev'] = time_since_prev(inspections,date_col='ACTUAL_END_DATE',
                                                    group_col='PGM_SYS_ID')
    inspections = merge_events(inspections,time_since_col='time_since_prev',days_thresh=insp_event_thresh)

    # Step 3: Load violation data
    violations_path = os.path.join(data_path,'ICIS-Air','ICIS-AIR_VIOLATION_HISTORY.csv')
    violations = load_violations(violations_path,start_date=start_date,end_date=end_date)

    # Step 4: Clean violation data
    violations['time_since_prev'] = time_since_prev(violations,date_col='VIOL_DATE',group_col='PGM_SYS_ID')
    violations = merge_events(violations,time_since_col='time_since_prev',days_thresh=insp_event_thresh)

    # Step 5: Create the links! 
    new_insp = link_viol_insp(inspections,violations,past_thresh=0,future_thresh=insp_viol_thresh_days)

    # Step 6: Save the linked inspections/violations to file. 
    insp_fname = 'processed_inspections_futurethresh_'+str(insp_viol_thresh_days)+'.csv'
    new_insp.to_csv(os.path.join(data_path,insp_fname))

    # Test linking percentage
    from tests import TEST_linking_percentage
    TEST_linking_percentage(violations,new_insp,log)

    log.write('Inspections linked to violations.\n')
    log.close()




