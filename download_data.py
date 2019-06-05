#!/usr/bin/env python3

import os, stat
import urllib.request
import zipfile
import shutil
import requests

def unzip_file(path_to_file,destination,new_name):
    """Function for unzipping a zip file. 
    
    PARAMETERS:
    **********

    INPUTS: 
    path_to_file = file_path/file_name to zip file. 
    destionation = destination directory for zip file's contents. 

    OUTPUTS: 
    None
    """
    
    zip_ref = zipfile.ZipFile(path_to_file, 'r')
    zip_ref.extractall(destination)
    zip_ref.close()

    return


def make_directory(dir_path): 
    try: 
        os.mkdir(dir_path)
    except FileExistsError: 
        print('directory already exists: '+dir_path)
    return None


def download_file_http(url,final_dest,final_name): 
    r = requests.get(url)
    with open(os.path.join(final_dest,final_name), 'wb') as f:
        f.write(r.content)
    return final_dest,final_name


def download_file_ftp(url,final_dest,final_name):
    urllib.request.urlretrieve(url, os.path.join(final_dest,final_name))
    return final_dest,final_name


if __name__=='__main__': 

    from external_variables import data_path
    from tests import TEST_ICISAir_filenames, TEST_ECHO_filenames
    from tests import TEST_FRS_filenames, TEST_NEI_filenames
    from tests import TEST_ICISAir_columns, TEST_ECHO_columns
    from tests import TEST_FRS_columns, TEST_NEI_columns

    log = open('logfile.txt','a')

    # Download ICIS-Air data
    icis_path = os.path.join(data_path, 'ICIS-Air')
    make_directory(icis_path)

    url = 'https://echo.epa.gov/files/echodownloads/ICIS-AIR_downloads.zip'
    file_path, file_name = download_file_http(url, icis_path, 'ICIS-Air.zip')
    unzip_file(os.path.join(file_path, file_name), file_path, 'ICIS-Air')
    os.remove(os.path.join(file_path, file_name))

    TEST_ICISAir_filenames(icis_path,log)
    TEST_ICISAir_columns(log)

    # Download ECHO facility data
    echo_path = os.path.join(data_path, 'ECHO')
    make_directory(echo_path)

    url = 'https://echo.epa.gov/files/echodownloads/echo_exporter.zip'
    file_path, file_name = download_file_http(url, echo_path, 'ECHO.zip')
    unzip_file(os.path.join(file_path, file_name), file_path, 'ECHO')
    os.remove(os.path.join(file_path, file_name))

    url = 'https://echo.epa.gov/system/files/echo_exporter_columns_02282019.xlsx'
    file_path, file_name = download_file_http(
        url, echo_path, 'echo_exporter_columns_02282019.xlsx')

    TEST_ECHO_filenames(echo_path,log)
    TEST_ECHO_columns(log)

    # Download FRS data
    frs_path = os.path.join(data_path, 'FRS')
    make_directory(frs_path)

    url = 'https://echo.epa.gov/files/echodownloads/frs_downloads.zip'
    file_path, file_name = download_file_http(url, frs_path, 'FRS.zip')
    unzip_file(os.path.join(file_path, file_name), file_path, 'FRS')
    os.remove(os.path.join(file_path, file_name))

    TEST_FRS_filenames(frs_path,log)
    TEST_FRS_columns(log)

    # Download NEI data
    nei_path = os.path.join(data_path, 'NEI')
    make_directory(nei_path)

    # For 2008
    url = 'ftp://newftp.epa.gov/air/nei/2008/data_summaries/2008neiv3_facility.zip'
    file_path, file_name = download_file_ftp(url, nei_path, 'NEI.zip')
    unzip_file(os.path.join(file_path, file_name), file_path, 'NEI')
    os.remove(os.path.join(file_path, file_name))
    os.rename(os.path.join(file_path, '2008neiv3_facility', '2008neiv3_facility.csv'),
            os.path.join(file_path, '2008neiv3_facility.csv'))
    shutil.rmtree(os.path.join(file_path, '2008neiv3_facility'))

    # For 2011
    url = 'ftp://newftp.epa.gov/air/nei/2011/data_summaries/2011v2/2011neiv2_facility.zip'
    file_path, file_name = download_file_ftp(url, nei_path, 'NEI.zip')
    unzip_file(os.path.join(file_path, file_name), file_path, 'NEI')
    os.remove(os.path.join(file_path, file_name))

    # For 2014
    url = 'ftp://newftp.epa.gov/air/nei/2014/data_summaries/2014v2/2014neiv2_facility.zip'
    file_path, file_name = download_file_ftp(url, nei_path, 'NEI.zip')
    unzip_file(os.path.join(file_path, file_name), file_path, 'NEI')
    os.remove(os.path.join(file_path, file_name))

    TEST_NEI_filenames(nei_path,log)
    TEST_NEI_columns(log)

    log.write('Data download complete.\n')
    log.close()
