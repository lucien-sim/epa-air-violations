#!/usr/bin/env python3

import os
import pandas as pd

# LIST OF TESTS: 

# 1. Verify names of downloaded files
def TEST_ICISAir_filenames(icis_path,log): 
    
    correct_fnames = ['ICIS-AIR_FACILITIES.csv','ICIS-AIR_PROGRAMS.csv',
                      'ICIS-AIR_FCES_PCES.csv','ICIS-AIR_PROGRAM_SUBPARTS.csv',
                      'ICIS-AIR_FORMAL_ACTIONS.csv','ICIS-AIR_STACK_TESTS.csv',
                      'ICIS-AIR_INFORMAL_ACTIONS.csv','ICIS-AIR_TITLEV_CERTS.csv',
                      'ICIS-AIR_POLLUTANTS.csv','ICIS-AIR_VIOLATION_HISTORY.csv']
    real_fnames = os.listdir(icis_path)
    missing_files = [fname for fname in correct_fnames if fname not in real_fnames]
    
    message = "The following ICIS-Air files are missing: "+", ".join(missing_files)
    try: 
        assert not missing_files, message
    except AssertionError as e:
        log.write(e)


def TEST_ECHO_filenames(echo_path,log): 
    
    correct_fnames = ['echo_exporter_columns_02282019.xlsx', 'ECHO_EXPORTER.csv']
    real_fnames = os.listdir(echo_path)
    missing_files = [fname for fname in correct_fnames if fname not in real_fnames]
    
    message = "The following ECHO files are missing: "+", ".join(missing_files)
    try: 
        assert not missing_files, message
    except AssertionError as e:
        log.write(e)


def TEST_FRS_filenames(frs_path,log): 
    
    correct_fnames = ['FRS_NAICS_CODES.csv','FRS_FACILITIES.csv',
                      'FRS_SIC_CODES.csv','FRS_PROGRAM_LINKS.csv']
    real_fnames = os.listdir(frs_path)
    missing_files = [fname for fname in correct_fnames if fname not in real_fnames]
    
    message = "The following FRS files are missing: "+", ".join(missing_files)
    try: 
        assert not missing_files, message
    except AssertionError as e:
        log.write(e)


def TEST_NEI_filenames(nei_path,log): 
    
    correct_fnames = ['2014v2facilities.csv', '2011neiv2_facility.csv', 
                      '2008neiv3_facility.csv']
    real_fnames = os.listdir(nei_path)
    missing_files = [fname for fname in correct_fnames if fname not in real_fnames]
    
    message = "The following NEI files are missing: "+", ".join(missing_files)
    try: 
        assert not missing_files, message
    except AssertionError as e:
        log.write(e)


# 2. Verify column names in downloaded files
def TEST_ICISAir_columns(log): 
    
    # ICIS-AIR_FCES_PCES.csv
    correct_columns = ['PGM_SYS_ID', 'ACTIVITY_ID', 'STATE_EPA_FLAG', 
                       'ACTIVITY_TYPE_CODE', 'ACTIVITY_TYPE_DESC', 
                       'COMP_MONITOR_TYPE_CODE', 'COMP_MONITOR_TYPE_DESC', 
                       'ACTUAL_END_DATE', 'PROGRAM_CODES'] 

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'ICIS-Air','ICIS-AIR_FCES_PCES.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in ICIS-AIR_FCES_PCES.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)
    
    # ICIS-AIR_VIOLATION_HISTORY.csv
    correct_columns = ['PGM_SYS_ID', 'ACTIVITY_ID', 'AGENCY_TYPE_DESC', 'STATE_CODE',
                       'AIR_LCON_CODE', 'COMP_DETERMINATION_UID', 'ENF_RESPONSE_POLICY_CODE',
                       'PROGRAM_CODES', 'PROGRAM_DESCS', 'POLLUTANT_CODES', 'POLLUTANT_DESCS',
                       'EARLIEST_FRV_DETERM_DATE', 'HPV_DAYZERO_DATE', 'HPV_RESOLVED_DATE']

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'ICIS-Air','ICIS-AIR_VIOLATION_HISTORY.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in ICIS-AIR_VIOLATION_HISTORY.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)
    
    # ICIS-AIR_FACILITIES.csv
    correct_columns = ['PGM_SYS_ID', 'REGISTRY_ID', 'FACILITY_NAME', 'STREET_ADDRESS', 'CITY',
                       'COUNTY_NAME', 'STATE', 'ZIP_CODE', 'EPA_REGION', 'SIC_CODES',
                       'NAICS_CODES', 'FACILITY_TYPE_CODE', 'AIR_POLLUTANT_CLASS_CODE',
                       'AIR_POLLUTANT_CLASS_DESC', 'AIR_OPERATING_STATUS_CODE',
                       'AIR_OPERATING_STATUS_DESC', 'CURRENT_HPV', 'LOCAL_CONTROL_REGION_CODE',
                       'LOCAL_CONTROL_REGION_NAME']

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'ICIS-Air','ICIS-AIR_FACILITIES.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in ICIS-AIR_FACILITIES.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)


def TEST_ECHO_columns(log): 
    
    # ECHO_EXPORTER.csv
    correct_columns = ['REGISTRY_ID','FAC_NAME','FAC_STREET','FAC_CITY','FAC_STATE',
                       'FAC_ZIP','FAC_COUNTY','FAC_FIPS_CODE','FAC_EPA_REGION',
                       'FAC_INDIAN_CNTRY_FLG','FAC_FEDERAL_FLG','FAC_US_MEX_BORDER_FLG',
                       'FAC_CHESAPEAKE_BAY_FLG','FAC_NAA_FLAG','FAC_LAT','FAC_LONG',
                       'FAC_MAP_ICON','FAC_COLLECTION_METHOD','FAC_REFERENCE_POINT',
                       'FAC_ACCURACY_METERS','FAC_DERIVED_TRIBES','FAC_DERIVED_HUC',
                       'FAC_DERIVED_WBD','FAC_DERIVED_STCTY_FIPS','FAC_DERIVED_ZIP',
                       'FAC_DERIVED_CD113','FAC_DERIVED_CB2010','FAC_PERCENT_MINORITY',
                       'FAC_POP_DEN','FAC_MAJOR_FLAG','FAC_ACTIVE_FLAG','FAC_MYRTK_UNIVERSE',
                       'FAC_INSPECTION_COUNT','FAC_DATE_LAST_INSPECTION',
                       'FAC_DAYS_LAST_INSPECTION','FAC_INFORMAL_COUNT',
                       'FAC_DATE_LAST_INFORMAL_ACTION','FAC_FORMAL_ACTION_COUNT',
                       'FAC_DATE_LAST_FORMAL_ACTION','FAC_TOTAL_PENALTIES',
                       'FAC_PENALTY_COUNT','FAC_DATE_LAST_PENALTY','FAC_LAST_PENALTY_AMT',
                       'FAC_QTRS_WITH_NC','FAC_PROGRAMS_WITH_SNC','FAC_COMPLIANCE_STATUS',
                       'FAC_SNC_FLG','FAC_3YR_COMPLIANCE_HISTORY','AIR_FLAG','NPDES_FLAG',
                       'SDWIS_FLAG','RCRA_FLAG','TRI_FLAG','GHG_FLAG','AIR_IDS',
                       'CAA_PERMIT_TYPES','CAA_NAICS','CAA_SICS','CAA_EVALUATION_COUNT',
                       'CAA_DAYS_LAST_EVALUATION','CAA_INFORMAL_COUNT',
                       'CAA_FORMAL_ACTION_COUNT','CAA_DATE_LAST_FORMAL_ACTION',
                       'CAA_PENALTIES','CAA_LAST_PENALTY_DATE','CAA_LAST_PENALTY_AMT',
                       'CAA_QTRS_WITH_NC','CAA_COMPLIANCE_STATUS','CAA_HPV_FLAG',
                       'CAA_3YR_COMPL_QTRS_HISTORY','NPDES_IDS','CWA_PERMIT_TYPES',
                       'CWA_COMPLIANCE_TRACKING','CWA_NAICS','CWA_SICS',
                       'CWA_INSPECTION_COUNT','CWA_DAYS_LAST_INSPECTION',
                       'CWA_INFORMAL_COUNT','CWA_FORMAL_ACTION_COUNT',
                       'CWA_DATE_LAST_FORMAL_ACTION','CWA_PENALTIES',
                       'CWA_LAST_PENALTY_DATE','CWA_LAST_PENALTY_AMT','CWA_QTRS_WITH_NC',
                       'CWA_COMPLIANCE_STATUS','CWA_SNC_FLAG','CWA_13QTRS_COMPL_HISTORY',
                       'CWA_13QTRS_EFFLNT_EXCEEDANCES','CWA_3_YR_QNCR_CODES','RCRA_IDS',
                       'RCRA_PERMIT_TYPES','RCRA_NAICS','RCRA_INSPECTION_COUNT',
                       'RCRA_DAYS_LAST_EVALUATION','RCRA_INFORMAL_COUNT',
                       'RCRA_FORMAL_ACTION_COUNT','RCRA_DATE_LAST_FORMAL_ACTION',
                       'RCRA_PENALTIES','RCRA_LAST_PENALTY_DATE','RCRA_LAST_PENALTY_AMT',
                       'RCRA_QTRS_WITH_NC','RCRA_COMPLIANCE_STATUS','RCRA_SNC_FLAG',
                       'RCRA_3YR_COMPL_QTRS_HISTORY','SDWA_IDS','SDWA_SYSTEM_TYPES',
                       'SDWA_INFORMAL_COUNT','SDWA_FORMAL_ACTION_COUNT','SDWA_COMPLIANCE_STATUS',
                       'SDWA_SNC_FLAG','TRI_IDS','TRI_RELEASES_TRANSFERS',
                       'TRI_ON_SITE_RELEASES','TRI_OFF_SITE_TRANSFERS','TRI_REPORTER_IN_PAST',
                       'FEC_CASE_IDS','FEC_NUMBER_OF_CASES','FEC_LAST_CASE_DATE',
                       'FEC_TOTAL_PENALTIES','GHG_IDS','GHG_CO2_RELEASES','DFR_URL',
                       'FAC_SIC_CODES','FAC_NAICS_CODES','FAC_DATE_LAST_INSPECTION_EPA',
                       'FAC_DATE_LAST_INSPECTION_STATE','FAC_DATE_LAST_FORMAL_ACT_EPA',
                       'FAC_DATE_LAST_FORMAL_ACT_ST','FAC_DATE_LAST_INFORMAL_ACT_EPA',
                       'FAC_DATE_LAST_INFORMAL_ACT_ST','FAC_FEDERAL_AGENCY','TRI_REPORTER',
                       'FAC_IMP_WATER_FLG','EJSCREEN_FLAG_US']

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'ECHO','ECHO_EXPORTER.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in ECHO_EXPORTER.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)


def TEST_FRS_columns(log): 
    
    # FRS_PROGRAM_LINKS.csv
    correct_columns = ['PGM_SYS_ACRNM', 'PGM_SYS_ID', 'REGISTRY_ID', 
                       'PRIMARY_NAME', 'LOCATION_ADDRESS', 'SUPPLEMENTAL_LOCATION', 
                       'CITY_NAME', 'COUNTY_NAME','FIPS_CODE', 'STATE_CODE', 
                       'STATE_NAME', 'COUNTRY_NAME', 'POSTAL_CODE']

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'FRS','FRS_PROGRAM_LINKS.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in FRS_PROGRAM_LINKS.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)


def TEST_NEI_columns(log): 
    
    # 2014v2facilities.csv
    correct_columns = ['eis_facility_site_id', 'program_system_code', 
                       'alt_agency_id', 'region_cd', 'st_usps_cd', 
                       'county_name', 'state_and_county_fips_code', 
                       'tribal_name', 'facility_site_name', 'naics_cd', 
                       'naics_description', 'facility_source_type', 
                       'latitude_msr', 'longitude_msr', 'location_address_text', 
                       'locality', 'addr_state_cd', 'address_postal_code', 
                       'emissions_operating_type', 'pollutant_cd', 'pollutant_desc', 
                       'total_emissions', 'uom', 'fips_state_code', 'company_name', 
                       'reporting_period']

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'NEI','2014v2facilities.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in 2014v2facilities.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)
    
    # 2011neiv2_facility.csv
    correct_columns = ['eis_facility_site_id', 'program_system_cd', 
                       'alt_agency_id', 'region_cd', 'st_usps_cd', 'county_name', 
                       'state_and_county_fips_code', 'tribal_name', 
                       'facility_site_name', 'naics_cd', 'facility_source_description', 
                       'facility_site_status_cd', 'latitude_msr', 'longitude_msr', 
                       'location_address_text', 'locality', 'addr_state_cd', 
                       'address_postal_code', 'emissions_op_type_code', 'pollutant_cd', 
                       'description', 'total_emissions', 'uom']

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'NEI','2011neiv2_facility.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in 2011neiv2_facility.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)
    
    # 2008neiv3_facility.csv
    correct_columns = ['eis_facility_site_id', 'program_system_cd', 'alt_agency_id', 
                       'region_cd', 'st_usps_cd', 'county_name', 
                       'state_and_county_fips_code', 'tribal_name', 'facility_site_name', 
                       'naics_cd', 'facility_source_description', 'facility_site_status_cd', 
                       'latitude_msr', 'longitude_msr', 'location_address_text', 'locality', 
                       'addr_state_cd', 'address_postal_code', 'emissions_op_type_code', 
                       'pollutant_cd', 'description', 'total_emissions', 'uom']

    from external_variables import data_path
    first_ten = pd.read_csv(os.path.join(data_path,'NEI','2008neiv3_facility.csv'),nrows=10)
    missing_columns = [col for col in correct_columns if col not in first_ten.columns]

    message = "The following columns are missing in 2008neiv3_facility.csv: "+\
                                                            ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)


# 3. Check % of links
def TEST_linking_percentage(violations,new_insp,log):
    total_violations = violations['PGM_SYS_ID'].count()
    linked_violations = new_insp['VIOL'].sum()
    percent_linked = linked_violations/total_violations*100
    message = "Less than 60% of violations were linked to inspections. Percentage linked = "+\
                                                                        "%.1f" % percent_linked
    try: 
        assert percent_linked>60, message
    except AssertionError as e:
        log.write(e)


# 4. Compare model performance in 2007, 2012, and 2018 to baselines
def TEST_model_performance(app_data_path,log): 
    results_df = pd.read_csv(os.path.join(app_data_path,'model_test_results.csv'),index_col=0)
    avg_performance = (results_df['model']/results_df['actual']).mean()
    message = 'Average performance is < 1.25. Value = ' + '%.2f' % avg_performance

    try: 
        assert avg_performance >= 1.25, message
    except AssertionError as e: 
        log.write(e)



# 5. Verify app data output format -> column names
def TEST_app_data_file(app_data_path,log):
    
    app_file = pd.read_csv(os.path.join(app_data_path,'web_app_data.csv'),index_col=0)
    
    # Check number of sources
    num_sources = app_file['SourceID'].count()
    message = 'Fewer than 150,000 source in app data file. Number of sources = '+str(num_sources)
    assert num_sources>150000, message

    # Check columns
    correct_columns = ['SourceID', 'RegistryID', 'Name', 'Address', 'City', 
                       'State', 'ZipCode', 'County', 'MajorEmitter', 'Latitude', 
                       'Longitude', 'Industry', 'Risk', 'Probability']
    missing_columns = [col for col in correct_columns if col not in app_file.columns]
    message = "The following columns are missing fromt the app data file: "+\
                                                                ', '.join(missing_columns)
    try: 
        assert not missing_columns, message
    except AssertionError as e:
        log.write(e)



if __name__=='__main__': 

    from external_variables import data_path, app_path
    app_data_path = os.path.join(app_path,'data')

    log = open('logfile.txt','a')

    # Test model performance on unseen years. 
    TEST_model_performance(app_data_path,log)

    # Verify that the app data file has the correct format and an 
    # appropriate number of facilities. 
    TEST_app_data_file(app_data_path,log)

    log.write('Final tests complete. If no assertion errors, ready to upload webapp to Heroku!\n')
    log.close()


    
    

