# Download Raw GEE and OSM Data for Large-Scale Multi-Modal Dataset Construction

This document outlines the steps required to download raw data from Google Earth Engine (GEE) and OpenStreetMap (OSM) for constructing a large-scale multi-modal dataset.

---

## 1. Database Setup

We use the following databases to store metadata and downloaded data:

- **Metadata Database**: SQLite3  
- **OSM Database**: SQLite3  
- **Image Patch Database**: MongoDB (optional)

For detailed information on setting up these databases, please refer to the [`database_builder.ipynb`](../tools/database_builder.ipynb) notebook in the `tools` folder.

---

## 2. Download GEE Data

Use the following script to download image patches from the USDA NAIP (DOQQ) dataset on Google Earth Engine. Adjust the parameters as needed.

```bash
#!/bin/bash
python tools/ee_image_downloader.py \
  --start_date "2021-8-1" \
  --end_date "2021-10-1" \
  --working_dir "./database" \
  --database_dir "metadata.db" \
  --log_file "img_download.log" \
  --mongo_uri "mongodb://localhost:27017/" \
  --drop_rate 0.5 \
  --service_account "scriptlogin@your-project.iam.gserviceaccount.com" \
  --credentials_file "credentials.json"
```

### GEE Data Download - Parameter Details

- **`--start_date` / `--end_date`**: Time range for the images to be downloaded (YYYY-MM-DD format).  
  
- **`--working_dir`**: Root directory for the metadata database, log file, and downloaded images (if `--mongo_uri` is not provided).  
  
- **`--database_dir`**: Relative or absolute path to the metadata database (joins with `working_dir` if relative).  
  
- **`--log_file`**: Path to the log file where download status and errors are recorded.  
  
- **`--mongo_uri`**: URI of the MongoDB server. If not provided, image patches are saved directly to `working_dir`.  
  
- **`--drop_rate`**: Probability of dropping an extracted patch to manage storage. For example, `0.5` means 50% of the patches are dropped.  
  
- **`--service_account`**: The service account used to access GEE. See [Service Accounts](https://developers.google.com/earth-engine/guides/service_account) for details.  
  
- **`--credentials_file`**: Path to the JSON file containing the service account credentials.

---

## 3. Download OSM Data

Use the following script to download OSM data (via the Overpass API) corresponding to your area of interest:

```bash
#!/bin/bash
python tools/osm_data_downloader.py \
  --meta_db_path "database/metadata.db" \
  --osm_db_path "database/osm.db" \
  --service_account "scriptlogin@your-project.iam.gserviceaccount.com" \
  --credentials_file "credentials.json" \
  --num_workers 50 \
  --prefetch_size 200
```

### OSM Data Download - Parameter Details

- **`--meta_db_path`**: Path to the metadata database.
  
- **`--osm_db_path`**: Path to the SQLite database where OSM data will be stored.

- **`--service_account`**: The service account used to authenticate requests to the GEE.
  
- **`--credentials_file`**: Path to the JSON file containing the service account credentials.
  
- **`--num_workers`**: Number of concurrent workers for downloading data.
  
- **`--prefetch_size`**: Batch size factor that multiplies by num_workers to define the total number of requests in each batch (i.e., $num\_workers \times prefetch\_size$).
