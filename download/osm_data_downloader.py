from OSMPythonTools.overpass import Overpass
from OSMPythonTools.overpass import overpassQueryBuilder

import pandas as pd
import geopandas as gpd
import numpy as np
import concurrent.futures
import sqlite3
import time
import json
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
import OSMPythonTools
import logging
from logging.handlers import RotatingFileHandler

from retry import retry
from google.api_core import retry as retry_google
import google
import ee
import gc

from ..annotation.utils.utils import get_area_nonearea_from_response_with_db

# create osm_data table
# TYPE: 0: background, 1: area, 2: nonearea
OSM_DATA_TABLE_INIT_CMD = """CREATE TABLE IF NOT EXISTS osm_data (
                            ID          INTEGER     PRIMARY KEY AUTOINCREMENT,
                            PATCH_NAME  TEXT        NOT NULL,
                            OSM_ID      INTEGER     NOT NULL,
                            TAGS        TEXT,   
                            TYPE        TINYINT     NOT NULL,
                            GEOMETRY    TEXT,
                            DATA_DATE   DATE               
                            )"""
# create osm_background table
OSM_BACKGROUND_TABLE_INIT_CMD = """CREATE TABLE IF NOT EXISTS osm_background (
                                    ID          INTEGER     PRIMARY KEY AUTOINCREMENT,
                                    OSM_ID      INTEGER     NOT NULL UNIQUE,
                                    TAGS        TEXT        NOT NULL
                                    )"""
                                    

SQL_INSERT_OSM_DATA="""insert into osm_data (PATCH_NAME, OSM_ID, TAGS, TYPE, GEOMETRY, DATA_DATE) values (?,?, ?, ?, ?, ?)"""

SQL_INSERT_BACKGROUND = "insert or ignore into osm_background (OSM_ID, TAGS) values (?,?)"
SQL_INSERT_OSM_DATA_BG = "insert into osm_data (PATCH_NAME, OSM_ID, TYPE) values (?,?,?)"

@retry_google.Retry()
def get_working_boundary_offset(center_pos, scale_x, scale_y, patch_size, osm_format=True):
    """
    Get the boundary by calculate the offset between the center to the working boundary.
    
    Args:
        center_pos (tuple): Center position of the boundary (lon, lat).
        scale_x (float): Unit variance for a pixel in x axis
        scale_y (float): Unit variance for a pixel in y axis
        patch_size (int): Patch size in pixels.
        osm_format (bool, optional): Whether to return the boundary in OSM format. Defaults to True.
                                    osm format:  [min_lat, min_lon, max_lat, max_lon]
                                    ee format:  [min_lon, min_lat, max_lon, max_lat]
    
    Returns:
        tuple: Boundary either in osm format or ee format.
    """
    
    offset_x = abs(scale_x * patch_size / 2)
    offset_y = abs(scale_y * patch_size / 2)
    
    x_min, x_max = center_pos[0] - offset_x, center_pos[0] + offset_x
    y_min, y_max = center_pos[1] - offset_y, center_pos[1] + offset_y
    
    if osm_format:
        return [y_min, x_min, y_max, x_max]
    else:
        return [x_min, y_min, x_max, y_max]
    
@retry_google.Retry()    
def get_working_boundary_buffer(center_pos, scale, patch_size, osm_format=True):
    """
    Get the boundary by using the ee.Geometry.buffer method

    Args:
        center_pos (tuple): Center position of the boundary (lon, lat).
        scale (float): Pixel resolusion in meters
        patch_size (int): Patch size in pixels
        osm_format (bool, optional): Whether to return the boundary in OSM format. Defaults to True.
                                    osm format:  [min_lat, min_lon, max_lat, max_lon]
                                    ee format:  [min_lon, min_lat, max_lon, max_lat]

    Returns:
        tuple: Boundary either in osm format or ee format.
    """
    point = ee.Geometry.Point(center_pos)
    bounds_ee = point.buffer(scale * patch_size / 2).bounds()
    bounds = bounds_ee.getInfo()
    bounds_np = np.array(bounds['coordinates'][0])
    x_min, x_max, y_min, y_max = bounds_np[:,0].min(), bounds_np[:,0].max(),\
                                 bounds_np[:,1].min(), bounds_np[:,1].max()
                                 
    if osm_format:
        return [y_min, x_min, y_max, x_max]
    else:
        return [x_min, y_min, x_max, y_max]

def check_column_exists(conn, table_name, column_name):
    
    c = conn.cursor()

    c.execute("PRAGMA table_info('{}');".format(table_name))
    columns = c.fetchall()
    
    exsits = any(column[1] == column_name for column in columns)
    return exsits

def check_column_exists_add(conn, table_name, column_name, column_def=''):

    c = conn.cursor()
    
    if not check_column_exists(conn, table_name, column_name):
        c.execute("ALTER TABLE {} ADD COLUMN {} {};".format(table_name, column_name, column_def))
        conn.commit()
        print("Added column {} to table {}".format(column_name, table_name))
    else:
        print("Column {} already exists in table {}".format(column_name, table_name))

def get_osm_working_boundary(center_pos, patch_dimension, download_time, patch_crs, scale):
    """
    Choose the boundary calculation function between `get_working_boundary_offset` and `get_working_boundary_buffer`.
    For the data downloaded before `2023-12-05` we use `get_working_boundary_offset`.

    Args:
        center_pos (tuple): (lat, lon) of the center of the patch
        patch_dimension (int): dimension of the patch in pixels
        download_time (str): timestamp of the data downloaded
        patch_crs (str): coordinate reference system of the patch
        scale (int): scale of the patch in meters per pixel

    Returns:
        tuple: (min_lat, min_lon, max_lat, max_lon) of the working boundary
    """

    boundary = get_working_boundary_buffer(center_pos, scale, patch_dimension)
        
    return boundary

def get_osm_data_2_insert(image_name, patch_name, osm_gdf, date_timestamp, data_type:int):
    data_2_insert = []
    for _, row in osm_gdf.iterrows():
        
        osm_id = row['id']
        patch_name_full = '/'.join([image_name, patch_name])
        tags = json.dumps(row['tags'], ensure_ascii=False)
        geometry = row['geometry'].wkt
        type_id = data_type
        data_date = date_timestamp.strftime("%Y-%m-%d")
        data_2_insert.append((patch_name_full, osm_id, tags, type_id, geometry, data_date))
        
    return data_2_insert

@retry(tries=3, delay=1, backoff=2)
def get_data_for_db(data, patch_meta, db_path):
    
    insert_data = []
    insert_background = []
    insert_osm_data = []
    
    id_metadata, patch_name, patch_dimension, patch_center_x, patch_center_y, patch_crs, image_name, download_time = data

    patch_center_x, patch_center_y = float(patch_center_x), float(patch_center_y)
    patch_dimension = int(patch_dimension)

    if patch_meta == None:
        OSMPythonTools.logger.error("No metadata found for patch: {}".format(id_metadata))
        return [], [], [], -1
    else:
        timestamp, scale = patch_meta[0], float(patch_meta[1])
    try:
        osm_working_area = get_osm_working_boundary((patch_center_x, patch_center_y), patch_dimension, download_time, patch_crs, scale)
    except:
        return [], [], [], -1
    # get the backgroud data
    overpass = Overpass('https://overpass.kumi.systems/api/')
    # aquire background information
    query = """
    is_in({lat},{lon})->.search;
    wr(pivot.search);
    out ids tags qt;
    """.format(lat=patch_center_y, lon=patch_center_x)
    try:
        response = overpass.query(query, settings={})
    except Exception as exc:
        OSMPythonTools.logger.error("Error in query background:")
        OSMPythonTools.logger.error(exc)
        del overpass
        return [], [], [], -1

    element_cnt = 0
    admin_level = []
    bg_information = []
    for element in response.elements():
        tags = element.tags()
        if tags == None:
            OSMPythonTools.logger.warning('encounter a None tag: {}'.format(tags))
            continue
        level = int(tags.get("admin_level", -1))
        admin_level.append(level)
        if level > 0:
            bg_information.append(tags)
        if 'boundary' in tags.keys() or 'admin_level' in tags.keys():
            id = element.id()
            patch_name_full = '/'.join([image_name, patch_name])
            insert_background.append((id, json.dumps(tags, ensure_ascii=False)))
            insert_data.append((patch_name_full, id, 0))
            element_cnt += 1
           
    # Sample the OSM data one month later
    struct_timestamp = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    date_timestamp = datetime(struct_timestamp.tm_year, struct_timestamp.tm_mon, struct_timestamp.tm_mday)
    date_timestamp += timedelta(days=30)
    format_time = date_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Query OSM data for the bounding box
    query = overpassQueryBuilder(bbox=osm_working_area,elementType=['way', 'relation'], includeGeometry=True)
    try:
        response = overpass.query(query, date=format_time)
    except Exception as exc:
        OSMPythonTools.logger.error("Error in query way and relation:".format(query))
        OSMPythonTools.logger.error(exc)
        del overpass
        return [], [], [], -1
    
    # Get the area and nonearea from the response
    try:
        osm_elements = get_area_nonearea_from_response_with_db(response.ways(), response.relations(), db_path, date_timestamp)
    except Exception as exc:
        OSMPythonTools.logger.error("Error in get_area_nonearea_from_response_with_db:")
        OSMPythonTools.logger.error(exc)
        del overpass
        return [], [], [], -1
    
    num_area = len(osm_elements['area'])
    num_nonearea = len(osm_elements['nonearea'])
    if num_area > 0:
        osm_area_gdf = gpd.GeoDataFrame(osm_elements['area'], crs='WGS84')
        data_2_insert = get_osm_data_2_insert(image_name, patch_name, osm_area_gdf, date_timestamp, 1)
        insert_osm_data.extend(data_2_insert)          
    
    if num_nonearea > 0:
        osm_nonarea_gdf = gpd.GeoDataFrame(osm_elements['nonearea'], crs='WGS84')
        data_2_insert = get_osm_data_2_insert(image_name, patch_name, osm_nonarea_gdf, date_timestamp, 2)
        insert_osm_data.extend(data_2_insert)          

    element_cnt += num_area + num_nonearea
    
    del overpass
    
    return insert_data, insert_background, insert_osm_data, element_cnt

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--meta_db_path', type=str, default='database/metadata.db')
    argparser.add_argument('--osm_db_path', type=str, default='database/osm.db')
    argparser.add_argument('--num_workers', type=int, default=50)
    argparser.add_argument('--prefetch_size', type=int, default=4)
    argparser.add_argument('--service_account', type=str, help='The service account to use for GEE authentication.')
    argparser.add_argument('--credentials_file', type=str, help='The file to save the GEE credentials.')
    args = argparser.parse_args()
    
    # set up logging
    handler = RotatingFileHandler(filename = 'osm_downloader.log', mode = 'a', maxBytes=10000000, backupCount=10)
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    OSMPythonTools.logger.addHandler(handler)
    # OSMPythonTools.logger.setLevel(logging.ERROR)
    
    # Initialize the Earth Engine API
    service_account = args.service_account
    credentials = ee.ServiceAccountCredentials(service_account, args.credentials_file)
    ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')

    # Connect to the metadata database
    conn_meta = sqlite3.connect(args.meta_db_path, timeout=60)

    # Connect to the OSM database
    conn_osm = sqlite3.connect(args.osm_db_path, timeout=240)

    # Download the water polygons
    # print('Loading water polygons...')
    # water_polygons = gpd.read_file(args.water_polygons_path)
    # print('Done !')

    c_osm = conn_osm.cursor()
    c_osm.execute(OSM_DATA_TABLE_INIT_CMD)
    c_osm.execute(OSM_BACKGROUND_TABLE_INIT_CMD)
    
    c_meta = conn_meta.cursor()
    
    batch_cnt = 1
    
    while True:
        
        try:
            c_meta.execute("select ID, NAME, DIMENSION, SAMPLE_CENTER_X, SAMPLE_CENTER_Y, CRS, IMAGE_NAME, DOWNLOAD_TIME \
                           from patch \
                           where NUM_MAP_ELEMENTS is NULL and ID > 8756578\
                           order by RANDOM() limit ?", 
                           (args.num_workers * args.prefetch_size,))
        except sqlite3.OperationalError:
            print("Database is locked, waiting for 10 seconds...")
            time.sleep(10)
            continue
        
        batch_data = c_meta.fetchall()
        if batch_data == None or len(batch_data) == 0:
            break
        
        patch_metas = []
        
        # Download the data
        for data in batch_data:

            c_image = conn_meta.cursor()
            try:
                c_image.execute("select TIMESTAMP, SCALE from image where NAME = '{}'".format(data[-2]))
            except sqlite3.OperationalError:
                print("Database is locked, waiting for 10 seconds...")
                time.sleep(10)
                break
            
            patch_meta = c_image.fetchone()
            patch_metas.append(patch_meta)
            
            # for data, patch_meta in tqdm(zip(batch_data, patch_metas), total=len(patch_metas)):
            #     results = get_data_for_db(data, patch_meta)
            #     insert_data, insert_background, insert_osm_data, element_cnt = results
           
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_data = {executor.submit(get_data_for_db, data, patch_meta, args.osm_db_path): data for data, patch_meta in zip(batch_data, patch_metas)}
            for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(future_to_data), desc='Downloading OSM data batch {}'.format(batch_cnt)):
                insert_data, insert_background, insert_osm_data, element_cnt = future.result()
                
                data = future_to_data.pop(future)
                
                if element_cnt == -1:
                    continue
                
                if len(insert_data) > 0:
                    c_osm.executemany(SQL_INSERT_OSM_DATA_BG, insert_data)
                
                if len(insert_background) > 0:
                    c_osm.executemany(SQL_INSERT_BACKGROUND, insert_background)
                
                if len(insert_osm_data) > 0:
                    c_osm.executemany(SQL_INSERT_OSM_DATA, insert_osm_data)
                
                c_meta.execute("update patch set NUM_MAP_ELEMENTS = {} where ID = {}".format(element_cnt, data[0]))
                conn_meta.commit()
                conn_osm.commit()
        batch_cnt += 1         
        gc.collect()

    conn_meta.close()
    conn_osm.close()
    print("Done !")
