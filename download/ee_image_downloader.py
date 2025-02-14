from google.api_core import retry
from retry import retry as _retry
import google

import argparse
import ee
import os
import numpy as np
import requests
import shutil
import time
import datetime
import sqlite3
import sys
from PIL import Image
from tqdm import tqdm
import concurrent.futures

import geopandas as gpd
import pandas as pd
from shapely import Point

from random import random
from shapely import Polygon, MultiPoint

import logging
from logging.handlers import RotatingFileHandler


# TODO: the following are the example parameters for downloading images from GEE

COLLECTION_TABLE_INIT_CMD ="""
                            CREATE TABLE IF NOT EXISTS collection(
                            ID      INTEGER     PRIMARY KEY AUTOINCREMENT,
                            NAME    TEXT        NOT NULL UNIQUE 
                        )"""
IMAGE_TABLE_INIT_CMD ="""
                        CREATE TABLE IF NOT EXISTS image(
                        ID          INTEGER     PRIMARY KEY AUTOINCREMENT,
                        NAME        TEXT        NOT NULL UNIQUE,
                        YEAR        INTEGER,
                        VERSION     INTEGER,
                        TIMESTAMP   INTEGER     NOT NULL,
                        FOOTPRINT   TEXT        NOT NULL,
                        ASSET_SIZE  INTEGER,
                        BANDS       TEXT,
                        SCALE       REAL        NOT NULL,
                        DIMENSION_H INTEGER,
                        DIMENSION_W INTEGER,
                        CRS TEXT    NOT NULL,
                        COLLECTION  TEXT        NOT NULL,
                        SAMPLE_INTERVAL FLOAT
                    )"""
PATCH_TABLE_INIT_CMD ="""
                        CREATE TABLE IF NOT EXISTS patch(
                        ID              INTEGER     PRIMARY KEY AUTOINCREMENT,
                        NAME            TEXT        NOT NULL,
                        IND_X           INTEGER,
                        IND_Y           INTEGER,
                        DIMENSION       INTEGER     NOT NULL,
                        SAMPLE_CENTER_X REAL        NOT NULL,
                        SAMPLE_CENTER_Y REAL        NOT NULL,
                        BANDS           TEXT        NOT NULL,
                        CRS             TEXT        NOT NULL,
                        FILE_FORMAT     TEXT        NOT NULL,
                        ASSET_SIZE      INTEGER,
                        DOWNLOAD_TIME   DATETIME    NOT NULL,
                        IMAGE_NAME      TEXT        NOT NULL,
                        NUM_MAP_ELEMENTS INTEGER    DEFAULT NULL,
                        NUM_ANNOTATIONS INTEGER    DEFAULT NULL,
                        FOREIGN KEY (IMAGE_NAME) REFERENCES image(NAME)
                    )"""


# set the scale property to the image collection
def get_image_scale(image):
    bands = image.bandNames()
    scales = bands.map(lambda b: image.select([b]).projection().nominalScale())
    scale = ee.Algorithms.If(
        scales.distinct().size().gt(1),
        ee.Number(-1),
        scales.get(0),
    )
    image = image.set("NOMINAL_SCALE", scale)
    return image

@_retry(tries=5, delay=1, backoff=2)
def get_patch(coords, image, request_fmt='PNG', scale=0.6, patch_size=448, features=None, vis_option=None):
    
    rand_delay = random()*0.5
    time.sleep(rand_delay)
    point = ee.Geometry.Point(coords)
    region = point.buffer(patch_size / 2 * scale).bounds()    
    
    request_dict = dict(
        region=region,
        dimensions=f"{patch_size}x{patch_size}",
        bands=features,
        ranges=vis_option['visualizationOptions']['ranges']
        )
    
    if request_fmt.lower() in ['png', 'jpg']:
        request_fmt = request_fmt.lower()
        request_dict['format'] = request_fmt
        url = image.getThumbURL(request_dict)
    else:
        assert request_fmt in ['GEO_TIFF', 'NPY', 'ZIPPED_GEO_TIFF'], f"Unsupported format: {request_fmt}"
        request_dict['format'] = request_fmt
        url = image.getDownloadURL(request_dict)
        
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()   
        
    return r 

def get_sample_patch_center(bounds, offset_x, offset_y):
    """_summary_

    Args:
        bounds (GeoJson): The footprint of the image region
        offset_x (float): _description_
        offset_y (float): _description_

    Returns:
        list: the list that contains the coordinates of the patch centers
    """
    shapely_roi = Polygon(bounds['coordinates'])
    lonmin, latmin, lonmax, latmax = shapely_roi.bounds
    # construct rectangle of points
    x, y = np.round(np.meshgrid(np.arange(lonmin, lonmax, offset_x),
                                np.arange(latmin, latmax, offset_y)),
                                7)
    points = MultiPoint(list(zip(x.flatten(),y.flatten())))

    # validate each point falls inside shapes
    valid_points=points.intersection(shapely_roi)
    valid_points = [p for p in valid_points.geoms]
    
    return valid_points

def get_sample_patch_center_gpd(bounds, offset_x, offset_y, drop_rate=0.0):
    """_summary_

    Args:
        bounds (GeoJson): The footprint of the image region
        offset_x (float): _description_
        offset_y (float): _description_
        drop_rate (float, optional): The rate of dropping some points. Defaults to 0.0.

    Returns:
        geopandas.GeoDataFrame: its columns are ['x_index', 'y_index', 'geometry(shapely.Point)']
    """
    
    assert 0 <= drop_rate < 1, "drop_rate should be between 0 and 1"
    
    shapely_roi = Polygon(bounds['coordinates'])
    lonmin, latmin, lonmax, latmax = shapely_roi.bounds
    # construct rectangle of points
    x, y = np.round(np.meshgrid(np.arange(lonmin, lonmax, offset_x),
                                np.arange(latmin, latmax, offset_y)),
                                7)
    x_ind, y_ind = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    points = list(map(Point, list(zip(x.flatten(),y.flatten()))))
    pos = list(zip(x_ind.flatten(),y_ind.flatten()))

    gpd_points = gpd.GeoDataFrame(pos, columns=['x', 'y'], geometry=points, crs='EPSG:4326')
    valid_gpd_points=gpd_points.clip(shapely_roi)
    
    # drop some points randomly with the probability of drop_rate
    if drop_rate > 0:
        valid_gpd_points = valid_gpd_points.sample(frac=1-drop_rate)
    
    return valid_gpd_points
    
@retry.Retry()
def get_sample_patch_offsets(image, patch_intervals=1, patch_size=448, scale=0.6):
    
    image_centroid = image.geometry().centroid()
    ref_roi = image_centroid.buffer(patch_size / 2 * scale).bounds()    
    ref_roi_info = ref_roi.getInfo()
    ref_geometry = np.array(ref_roi_info['coordinates'][0])
    lon_min, lon_max = ref_geometry[:, 0].min(), ref_geometry[:, 0].max()
    lat_min, lat_max = ref_geometry[:, 1].min(), ref_geometry[:, 1].max()
    offset_x = lon_max - lon_min
    offset_y = lat_max - lat_min
    
    return offset_x * patch_intervals, offset_y * patch_intervals


@retry.Retry()
def get_info_retry(ee_obj):
    
    return ee_obj.getInfo()

@retry.Retry()
def get_image_infos(img):
    """_summary_

    Args:
        img (ee.Image): _description_

    Returns:
        _type_: _description_
    """
    
    # delay a random time in case too many requests at the same time
    # TODO: use global definition
    # rand_delay = random()*1
    # time.sleep(rand_delay)
    
    keys = img.propertyNames()
    values = keys.map(lambda p: img.get(p))
    props = ee.Dictionary.fromLists(keys, values)
    img_infos = props.getInfo()
    bands = img_infos['system:bands']
    multiband_dimension = []
    multiband_crs = []

    for band in bands:
        band_info = img_infos['system:bands'][band]
        multiband_dimension.append(band_info['dimensions'])
        multiband_crs.append(band_info['crs'])
    crs_same_flag, dimension_same_flag = True, True
    for crs in multiband_crs:
        if crs != multiband_crs[0]:
            crs_same_flag = False
            break
    for dimensions in multiband_dimension:
        if dimensions != multiband_dimension[0]:
            dimension_same_flag = False
            break
    multiband_dimension = multiband_dimension[0] if dimension_same_flag else [-1, -1]
    multiband_crs = multiband_crs[0] if crs_same_flag else str(multiband_crs)

    name = img_infos['system:index']
    img_id = img_infos['system:id']
    if name != img_id.split('/')[-1]:
        logger.warning("system:index and system:id are not consistent for image: %s, its index is: %s. Using id instead", img_id, name)
        name = img_id.split('/')[-1]
    collection_name = '/'.join(img_id.split('/')[:-1])
    timestamp = time.gmtime(img_infos['system:time_start']//1000)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", timestamp)
    footprint = img_infos['system:footprint']
    asset_size = img_infos['system:asset_size']
    year = img_infos.get('year', -1)
    version = img_infos.get('system:version', -1)
    scale = img_infos['NOMINAL_SCALE'] if isinstance(img_infos['NOMINAL_SCALE'], float) else -1
    bands = list(bands.keys())
    
    infos_2_save = dict(NAME=name,
                        YEAR=year,
                        VERSION=version,
                        TIMESTAMP=timestamp,
                        FOOTPRINT=footprint,
                        ASSET_SIZE=asset_size,
                        BANDS=str(bands),
                        SCALE=scale,
                        DIMENSION_H=multiband_dimension[0],
                        DIMENSION_W=multiband_dimension[1],
                        CRS=multiband_crs,
                        COLLECTION=collection_name)
    return infos_2_save

def get_patch_and_save(point, mongo_client):
    """_summary_

    Args:
        point (Series): keys: ['x', 'y', 'geometry', 'selected_img', 'img_infos', 'working_dir_image', 'patch_format', 'scale', 'patch_size', 'working_crs', 'features', 'vis_option']

    Returns:
        _type_: _description_
    """
    selected_img, img_infos, working_dir_image, patch_format, scale, patch_size, working_crs, features, vis_option = \
    point['selected_img'], eval(point['img_infos']), point['working_dir_image'], point['patch_format'], point['scale'], point['patch_size'], point['working_crs'], eval(point['features']), eval(point['vis_option'])
    response = get_patch((point['geometry'].xy[0][0], point['geometry'].xy[1][0]), selected_img, patch_format, scale, patch_size, features, vis_option)
    timestamp = time.localtime()
    download_time = time.strftime("%Y-%m-%d %H:%M:%S", timestamp)
    # save the patch image
    # let's define the funcions' look-up-table for easy access
    # TODO: add more functions related to other image formats
    if mongo_client is not None:
        dataset = point['dataset']
        collection = img_infos['NAME']
        mongo_collection = mongo_client[dataset][collection]
    
    def save_response(response, work_dir, name):
        ext_lut = dict(JPG='.jpg',
                       PNG='.png',
                       GEO_TIFF='.tif',
                       NPY='.npy')
        
        ext = ext_lut[patch_format]
        
        save_path = os.path.join(work_dir, name+ext)
        with open(save_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        
        asset_size = os.path.getsize(save_path)
        return asset_size
    
    def save_response_mongo(response, name, mongo_collection):
        ext_lut = dict(JPG='.jpg',
                       PNG='.png',
                       GEO_TIFF='.tif',
                       NPY='.npy')
        
        ext = ext_lut[patch_format]
        mongo_collection.insert_one({'name': name, 'ext': ext, 'patch': response.content})
        # get the size of the saved file
        asset_size = len(response.content)
        return asset_size

    # get all the metadata for this patch
    ind_x, ind_y = point['x'], point['y']
    name = '{}_{}'.format(ind_x, ind_y)
    dimension = patch_size
    center_x, center_y = point['geometry'].xy[0][0], point['geometry'].xy[1][0]
    bands = ''.join(features)
    crs = working_crs
    file_format=patch_format

    # save the image data
    if mongo_client is None:
        asset_size = save_response(response, working_dir_image, name)
    else:
        asset_size = save_response_mongo(response, name, mongo_collection)

    patch_infos=dict(NAME=name,
                     IND_X=ind_x,
                     IND_Y=ind_y,
                     DIMENSION=dimension,
                     SAMPLE_CENTER_X=center_x,
                     SAMPLE_CENTER_Y=center_y,
                     BANDS=bands,
                     CRS=crs,
                     FILE_FORMAT=file_format,
                     ASSET_SIZE=asset_size,
                     DOWNLOAD_TIME=download_time,
                     IMAGE_NAME=img_infos['NAME'])

    return patch_infos

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download images from GEE and save them to disk.')
    parser.add_argument('--database', type=str, default='SQLite', help='The database to use for storing downloaded images.')
    parser.add_argument('--patch_format', type=str, default='PNG', help='The format of the downloaded patches.')
    parser.add_argument('--dataset', type=str, default='USDA/NAIP/DOQQ', help='The dataset to download images from.')
    parser.add_argument('--start_date', type=str, default='2022-3-29', help='The start date of the image collection.')
    parser.add_argument('--end_date', type=str, default='2022-4-5', help='The end date of the image collection. This is not inclusive.')
    parser.add_argument('--scale', type=float, default=0.6, help='The scale of the downloaded images in meters.')
    parser.add_argument('--features', type=str, default=','.join(['R', 'G', 'B']), help='The features to download.')
    parser.add_argument('--working_crs', type=str, default='EPSG:4326', help='The CRS of the downloaded images.')
    parser.add_argument('--patch_size', type=int, default=448, help='The size of the downloaded patches in pixels.')
    parser.add_argument('--vis_option', type=str, default=str({'visualizationOptions': {'ranges': [{'min': 0, 'max': 255}]}}), help='The visualization options for normalizing the image bands.')
    parser.add_argument('--n_workers', type=int, default=50, help='The number of workers for asynchronous downloading.')
    parser.add_argument('--kernel_shape', type=str, default=str([448, 448]), help='The shape of the patches expected by the model.')
    parser.add_argument('--working_dir', type=str, default='./database', help='The folder to save all images and database files.')
    parser.add_argument('--database_dir', type=str, default='metadata.db', help='The folder to save the downloaded database files.')
    parser.add_argument('--starting_ind', type=int, default=0, help='The starting index of the downloaded.')
    parser.add_argument('--sample_interval', type=int, default=2, help='The sample interval between patches.')
    parser.add_argument('--log_file', type=str, default='downloader.log', help='The log file to save the downloading process.')
    parser.add_argument('--db_timeout', type=int, default=60, help='The timeout for the database connection in seconds.')
    parser.add_argument('--drop_rate', type=float, default=0.9, help='The rate of dropping some points.')
    parser.add_argument('--service_account', type=str, help='The service account to use for GEE authentication.')
    parser.add_argument('--credentials_file', type=str, help='The file to save the GEE credentials.')
    parser.add_argument('--mongo_uri', type=str, default=None, help='The URI of the MongoDB server. default: None. If specified, the downloaded images will be saved to the MongoDB server with the provided URI.')
    
    args = parser.parse_args()

    database = args.database
    patch_format = args.patch_format
    dataset = args.dataset
    start_date = args.start_date
    end_date = args.end_date
    scale = args.scale
    features = args.features.split(',')
    working_crs = args.working_crs
    patch_size = args.patch_size
    vis_option = eval(args.vis_option)
    n_workers = args.n_workers
    kernel_shape = eval(args.kernel_shape)
    working_dir = args.working_dir
    database_dir = args.database_dir if args.database_dir.startswith('/') \
                                    else os.path.join(working_dir, args.database_dir)
    starting_ind = args.starting_ind
    sample_interval = args.sample_interval
    log_file = args.log_file if args.log_file.startswith('/') \
                                    else os.path.join(working_dir, args.log_file)
    db_timeout = args.db_timeout
    drop_rate = args.drop_rate

    # check if the folder exists, otherwise create it
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    service_account = args.service_account
    credentials = ee.ServiceAccountCredentials(service_account, args.credentials_file)
    ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')

    logger = logging.getLogger('[img downloader]')
    # handler = logging.FileHandler(filename = 'downloader_test.log', mode = 'a')
    handler = RotatingFileHandler(filename = log_file, maxBytes=10000000, backupCount=9)
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    mongo_uri = args.mongo_uri
    mongo_client = None
    if mongo_uri is not None:
        logger.info("Using MongoDB server at {}".format(mongo_uri))
        from pymongo import MongoClient
        mongo_client = MongoClient(mongo_uri)
    
    # let log all the settings
    logger.info('Settings:')
    logger.info('database: {}'.format(database))
    logger.info('patch_format: {}'.format(patch_format))
    logger.info('dataset: {}'.format(dataset))
    logger.info('start_date: {}'.format(start_date))
    logger.info('end_date: {}'.format(end_date))
    logger.info('scale: {}'.format(scale))
    logger.info('features: {}'.format(features))
    logger.info('working_crs: {}'.format(working_crs))
    logger.info('patch_size: {}'.format(patch_size))
    logger.info('vis_option: {}'.format(vis_option))
    logger.info('n_workers: {}'.format(n_workers))
    logger.info('kernel_shape: {}'.format(kernel_shape))
    logger.info('working_dir: {}'.format(working_dir))
    logger.info('database_dir: {}'.format(database_dir))
    logger.info('starting_ind: {}'.format(starting_ind))
    logger.info('sample_interval: {}'.format(sample_interval))
    logger.info('log_file: {}'.format(log_file))
    logger.info('db_timeout: {}'.format(db_timeout))
    logger.info('drop_rate: {}'.format(drop_rate))
    logger.info('mongo_uri: {}'.format(mongo_uri))
    
    logger.info('Downloading task started!')

    # check if the start_date is earlier than the end_date
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    if end_date < start_date:
        logger.error('The start date is later than the end date!')
        sys.exit(1)

    # init the data base
    database_file = database_dir
    conn = sqlite3.connect(database_file, timeout=db_timeout)

    c = conn.cursor()
    c.execute(COLLECTION_TABLE_INIT_CMD)
    logger.info("collection table initialized")
    c.execute(IMAGE_TABLE_INIT_CMD)
    logger.info("image table initialized")
    c.execute(PATCH_TABLE_INIT_CMD)
    logger.info("patch table initialized")
    
    logger.info("Fetching COLLECTION {}. Date between {} and {}. Scale: {}".format(dataset, start_date, end_date, scale))
    
    # iter throught the start_date and end_date to download images with a backward step of 1 day
    date_list = [end_date - datetime.timedelta(days=x) for x in range(1, (end_date - start_date).days+1, 1)]
    
    # init progress bar
    pbar = tqdm(total=len(date_list))
    
    first_iter_flag = True
    while date_list:
        current_date = date_list.pop()
        current_date_str = current_date.strftime(date_format)
        logger.info("Fetching images on {}".format(current_date_str))
        
        download_collection = ee.ImageCollection(dataset).filterDate(current_date_str)
        download_collection = download_collection.map(get_image_scale)
        download_collection = download_collection.filter(ee.Filter.eq('NOMINAL_SCALE', scale))
        try:
            n_imgs = get_info_retry(download_collection.size())
        except Exception as exc:
            logger.error("Failed to get the size of the download COLLECTION {}, date {}. {}".format(dataset, current_date, str(exc)))
            continue
        logger.info('The download COLLECTION {} size is: {}. Data date: {}'.format(dataset, n_imgs, current_date_str))
        # insert current collection information
        try:
            c.execute("INSERT OR IGNORE INTO collection (NAME) VALUES ('{}')".format(dataset))
        except Exception as exc:
            logger.error('Inserting COLLECTION {} encounters an error! {}'.format(dataset, str(exc)))
        else:
            logger.info("COLLECTION {} inserted successfully".format(dataset))
            conn.commit()
        # create the folder of current image collection
        folder_name = '_'.join(dataset.split('/'))
        working_dir_collection = os.path.join(working_dir, folder_name)
        if not os.path.exists(working_dir_collection):
            os.makedirs(working_dir_collection)

        download_n_imgs = n_imgs - starting_ind if first_iter_flag else n_imgs

        selected_img_list = download_collection.toList(download_n_imgs, starting_ind) if first_iter_flag \
                       else download_collection.toList(download_n_imgs) 

        first_iter_flag = False
        failed_images = []

        valid_gpd_points = []
        
        if download_n_imgs == 0:
            pbar.update(1)
            continue
        
        progress_bar = tqdm(range(download_n_imgs), leave=False, desc='Preparing image patches for date {}'.format(current_date_str))
        for i in range(download_n_imgs):
            selected_img = selected_img_list.get(i)
            selected_img = ee.Image(selected_img)
            logger.info("Starting to prepare patches for the {} of {} IMAGE...".format(i+1, download_n_imgs))
            try:
                img_infos = get_image_infos(selected_img)
                offset_x, offset_y = get_sample_patch_offsets(selected_img, sample_interval, patch_size, scale)
                img_infos.update({'SAMPLE_INTERVAL':sample_interval})
            except Exception as exc:
                logger.error("Encountered error downloading {} of {} IMAGE. {}".format(i+1, download_n_imgs, str(exc)))
                failed_images.append(i)
                progress_bar.update(1)
                continue
            
            bounds = img_infos['FOOTPRINT']
            valid_gpd_points_single = get_sample_patch_center_gpd(bounds, offset_x, offset_y, drop_rate)
            
            if len(valid_gpd_points_single) == 0:
                logger.info("No valid sample points found for IMAGE {}. Skipping...".format(i+1))
                progress_bar.update(1)
                continue
            
            # insert selected image information into the database
            IMAGE_KEYS = ['NAME','YEAR','VERSION','TIMESTAMP','FOOTPRINT','ASSET_SIZE','BANDS','SCALE','DIMENSION_H','DIMENSION_W','CRS','COLLECTION','SAMPLE_INTERVAL']
            sql_insert_image="""INSERT OR IGNORE INTO image (NAME,YEAR,VERSION,TIMESTAMP,FOOTPRINT,ASSET_SIZE,BANDS,SCALE,DIMENSION_H,DIMENSION_W,CRS,COLLECTION,SAMPLE_INTERVAL) VALUES ("{}",{},{},"{}","{}",{},"{}",{},{},{},"{}","{}", {})""".format(*[img_infos[key] for key in IMAGE_KEYS])
            try:
                c.execute(sql_insert_image)
            except Exception as exc:
                logger.error('Inserting IMAGE {} encounters an error! {}'.format(img_infos['NAME'], str(exc)))
                progress_bar.update(1)
                continue
            else:
                logger.info("IMAGE {} inserted successfully".format(img_infos['NAME']))
                conn.commit()
            # create the folder for saving the image patches
            working_dir_image = os.path.join(working_dir_collection, img_infos['NAME'])
            if not os.path.exists(working_dir_image):
                os.makedirs(working_dir_image)

            # add the following columns to the valid_gpd_points_single: selected_img, img_infos, working_dir_image, patch_format, patch_size, working_crs, features
            valid_gpd_points_single['selected_img'] = selected_img
            valid_gpd_points_single['img_infos'] = str(img_infos)
            valid_gpd_points_single['working_dir_image'] = working_dir_image
            valid_gpd_points_single['patch_format'] = patch_format
            valid_gpd_points_single['scale'] = scale
            valid_gpd_points_single['vis_option'] = str(vis_option)
            valid_gpd_points_single['patch_size'] = patch_size
            valid_gpd_points_single['working_crs'] = working_crs
            valid_gpd_points_single['features'] = str(features)
            valid_gpd_points_single['dataset'] = folder_name
            valid_gpd_points.append(valid_gpd_points_single)
            progress_bar.update(1)
        success_cnt=0
        failed_patches = []
        if len(valid_gpd_points) == 0:
            logger.error("Encountered error getting valid sample points for date {}. Skipping...".format(current_date_str))
            continue
        # concatenate all the valid_gpd_points into a gpd.GeoDataFrame
        valid_gpd_points = pd.concat(valid_gpd_points)
        # We can use a with statement to ensure threads are cleaned up promptly
        logger.info("Starting to download patches of date {}...".format(current_date_str))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Start the load operations and mark each future with its URL
            future_to_patch = {executor.submit(get_patch_and_save, point, mongo_client): point for _, point in valid_gpd_points.iterrows()}
            patch_progress_bar = tqdm(total=len(valid_gpd_points), leave=False, desc='Downloading patches')
            for future in concurrent.futures.as_completed(future_to_patch):
                point = future_to_patch.pop(future)
                try:
                    patch_infos = future.result()
                except Exception as exc:
                    logger.error("Downloading a PATCH failed! {}".format(str(exc)))
                    failed_patches.append(str(point))
                else:
                    # store metadata into the database
                    sql_insert_patch="""
                            INSERT INTO patch (NAME,IND_X,IND_Y,DIMENSION,SAMPLE_CENTER_X,SAMPLE_CENTER_Y,BANDS,CRS,FILE_FORMAT,ASSET_SIZE,DOWNLOAD_TIME,IMAGE_NAME) 
                            VALUES ("{}",{},{},{},{},{},"{}","{}","{}",{},"{}","{}")
                                    """.format(patch_infos['NAME'],
                                                patch_infos['IND_X'],
                                                patch_infos['IND_Y'],
                                                patch_infos['DIMENSION'],
                                                patch_infos['SAMPLE_CENTER_X'],
                                                patch_infos['SAMPLE_CENTER_Y'],
                                                patch_infos['BANDS'],
                                                patch_infos['CRS'],
                                                patch_infos['FILE_FORMAT'],
                                                patch_infos['ASSET_SIZE'], 
                                                patch_infos['DOWNLOAD_TIME'],
                                                patch_infos['IMAGE_NAME'])

                    # we do not log success, as there may be too many records
                    try:
                        c.execute(sql_insert_patch)
                    except Exception as exc:
                        logger.error('Inserting PATCH {} of IMAGE {} encounters an error! {}'.format(patch_infos['NAME'], img_infos['NAME'], str(exc)))
                    else:
                        success_cnt+=1
                        conn.commit() 
                patch_progress_bar.update(1)
            conn.commit()    
            # log the fails
            num_fails = len(failed_patches)
            if num_fails>0:
                logger.error("Downloading PATCH of date {} encountered {} failures {} successes. Failed patches are:\r{}".format(current_date_str, num_fails, success_cnt, '\r-------------------\r'.join(failed_patches)))
            else:
                logger.info("Downloading PATCH of date {} all successed! {} PATCH in total".format(current_date_str, success_cnt))
            
        conn.commit()
        pbar.update(1)
    conn.commit()
    logger.info("Disconnecting dabase...")
    conn.close()
    logger.info('Downloading finished.')
    if mongo_client is not None:
        mongo_client.close()
        logger.info('MongoDB connection closed.')
    if len(failed_images) > 0:
        logger.error('Failed to start the downloading of the IMAGE indexes are: {}'.format(str(failed_images)))