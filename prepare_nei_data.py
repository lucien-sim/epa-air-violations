#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_NEI_data(poll_codes, yrs):

    def get_facilities_list(file_facilities_icis):
        facilities_icis = pd.read_csv(file_facilities_icis, dtype='str')
        facilities_icis = facilities_icis[['REGISTRY_ID', 'NAICS_CODES']].dropna(
            axis=0, subset=['REGISTRY_ID'])
        facilities_icis = facilities_icis[facilities_icis['REGISTRY_ID'].duplicated(
        ) == 0]
        return facilities_icis

    def add_NEI_IDs(X, file_program_links_frs):
        """Adds NEI IDs for all facilities (for which NEI IDs are available)."""
        program_links = pd.read_csv(file_program_links_frs, dtype='str')
        NEI_links = program_links[program_links['PGM_SYS_ACRNM'] == 'EIS']
        NEI_links = NEI_links.rename(
            columns={'PGM_SYS_ID': 'EIS_ID'}).drop('PGM_SYS_ACRNM', axis=1)
        X = X.merge(NEI_links, how='left', on='REGISTRY_ID')
        return X

    def add_emissions_from_yr(X, poll_codes, file_nei, nei_year='2014'):
        """Adds emissions estimates for each pollutant in poll_codes."""
        nei = pd.read_csv(file_nei, dtype='str')
        cols_for_merge = ['total_emissions']
        for code in poll_codes:
            for_merge = nei[nei['pollutant_cd'] == code][[
                'eis_facility_site_id']+cols_for_merge]
            for_merge[cols_for_merge] = for_merge[cols_for_merge].astype(float)
            rename_dict = {col: col+':'+code+':' +
                           str(nei_year) for col in cols_for_merge}
            for_merge = for_merge.rename(columns=rename_dict)
            X = X.merge(for_merge, how='left', left_on='EIS_ID',
                        right_on='eis_facility_site_id').drop('eis_facility_site_id', axis=1)
        return X

    # Get list of relevant facilities
    file_facilities_icis = os.path.join(
        data_path, 'ICIS-Air', 'ICIS-AIR_FACILITIES.csv')
    icis_facilities = get_facilities_list(file_facilities_icis)

    # Link REGISTRY_ID to NEI_ID
    file_program_links_frs = os.path.join(
        data_path, 'FRS', 'FRS_PROGRAM_LINKS.csv')
    icis_facilities = add_NEI_IDs(icis_facilities, file_program_links_frs)

    # Add 2014 NEI data
    if '2014' in yrs:
        file_nei14 = os.path.join(
            data_path, 'NEI', '2014v2facilities.csv')
        icis_facilities = add_emissions_from_yr(
            icis_facilities, poll_codes, file_nei14, nei_year='2014')

    # Add 2011 NEI data
    if '2011' in yrs:
        file_nei11 = os.path.join(
            data_path, 'NEI', '2011neiv2_facility.csv')
        icis_facilities = add_emissions_from_yr(
            icis_facilities, poll_codes, file_nei11, nei_year='2011')

    # Add 2008 NEI data
    if '2008' in yrs:
        file_nei08 = os.path.join(
            data_path, 'NEI', '2008neiv3_facility.csv')
        icis_facilities = add_emissions_from_yr(
            icis_facilities, poll_codes, file_nei08, nei_year='2008')

    # Remove duplicates introduced by left joins.
    icis_facilities = icis_facilities[icis_facilities['REGISTRY_ID'].duplicated(
    ) == False]

    return icis_facilities


def add_industry_nei(X):
    """Transformer for adding feature: industry of regulated facility"""
    from external_variables import naics_dict
    naics_lookup = pd.DataFrame({'FIRST_NAICS': list(naics_dict.keys()),
                                 'FAC_INDUSTRY': list(naics_dict.values())})
    X['FIRST_NAICS'] = X['NAICS_CODES'].apply(
        lambda x: str(x).split(' ')[0][0:2])
    X = X.merge(naics_lookup, how='left', on='FIRST_NAICS')
    X = X.drop('FIRST_NAICS', axis=1)
    X['FAC_INDUSTRY'] = X['FAC_INDUSTRY'].fillna('unknown')
    return X


def calc_primary_emissions(nei_data):
    """Function for calculating emissions for the primary pollutants for ALL facilities, normalized by 
    the mean emissions for all facilities in a given facility's industry. 
    """

    def get_primary_poll_for_industry(nei_data, yr):
        """Function to get 'primary pollutants' for each industry.
        'primary pollutants' are defined as the three pollutants that are highest, relative to the 
        corss-industry emission values. 
        """
        # Get mean emissions totals for each pollutant, for each industry.
        needed_cols = ['FAC_INDUSTRY'] + \
            [col for col in nei_data.columns if '2014' in col]
        mean_emiss = nei_data[needed_cols].groupby('FAC_INDUSTRY').mean()

        # Norm. emissions of each pollutant by dividing by the mean across all industries. Primary pollutants
        # for an industry are the those that have the largest emissoins relative to cross-industry means.
        primary_poll = {}
        mean_emiss_quant = mean_emiss.copy()
        for i, row in mean_emiss_quant.iterrows():
            mean_emiss_quant.loc[i,
                                 :] = mean_emiss_quant.loc[i, :]/mean_emiss.mean()
            primary_poll[i] = {'poll'+str(i+1): name.split(':')[1] for
                               i, name in enumerate(list(row.nlargest(3).index))}
        return primary_poll

    def calc_mean_emiss_by_industry(nei_data, years=['2008', '2011', '2014']):
        """Function for calculating mean emissions of each pollutant, for each industry"""
        mean_emiss_by_year = {}
        for year in years:
            needed_cols = ['FAC_INDUSTRY'] + \
                [col for col in nei_data.columns if year in col]
            mean_emiss = nei_data[needed_cols].groupby('FAC_INDUSTRY').mean()
            mean_emiss_by_year[year] = mean_emiss.rename(columns={col: col.split(':')[1] for col
                                                                  in mean_emiss.columns})
        return mean_emiss_by_year

    def add_primary_poll_cols(row, poll_num, year, primary_poll, mean_emiss):
        """Function for calculating emissions for the primary pollutants for a SINGLE facility, normalized by 
        the emissions for all facilities in the industry. 
        """
        poll_name = primary_poll[row['FAC_INDUSTRY']]['poll'+str(poll_num)]
        poll_val = row[':'.join(['total_emissions', poll_name, year])] / \
            mean_emiss[year].loc[row['FAC_INDUSTRY'], poll_name]
        return poll_val

    primary_poll = get_primary_poll_for_industry(nei_data, '2014')
    mean_emiss = calc_mean_emiss_by_industry(
        nei_data, years=['2008', '2011', '2014'])
    for year in ['2008', '2011', '2014']:
        for poll_num in range(1, 4):
            new_col = []
            for _, row in nei_data.iterrows():
                new_col.append(add_primary_poll_cols(
                    row, poll_num, year, primary_poll, mean_emiss))
            nei_data['poll'+str(poll_num)+'_'+year] = new_col

    return nei_data, primary_poll


if __name__ == '__main__':

    from external_variables import data_path

    log = open('logfile.txt','a')

    # Years of NEI data, pollutant codes. 
    yrs = ['2008', '2011', '2014']
    poll_codes = ['NOX', 'PM10-PRI', 'PM25-PRI','PMFINE', 'SO2', 'SO4', 'VOC', 'CO', 'NH3']

    # Retrieve NEI data for all pollutants, for all facilities, in all years. 
    # Place all that data in a single dataframe. 
    nei_data = get_NEI_data(poll_codes, yrs)

    # Add column for the facility's industry. 
    nei_data = add_industry_nei(nei_data)

    # Calculate normalized emissions of industry's primary pollutants, for each facility. 
    nei_data, primary_poll = calc_primary_emissions(nei_data)
    nei_data = nei_data[nei_data['REGISTRY_ID'].duplicated() == 0]

    # Save complete dataset to CSV. Also save dataframe that lists each industry's primary pollutants. 
    primary_poll_df = pd.DataFrame(primary_poll)
    primary_poll_df.to_csv(os.path.join(
        data_path, 'primary_pollutants_by_industry.csv'))
    nei_data.to_csv(os.path.join(
        data_path, 'processed_nei_emissions_by_facility.csv'))

    log.write('NEI data processing complete.\n')
    log.close()

