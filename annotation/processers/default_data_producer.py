from multiprocessing import Queue
from queue import PriorityQueue
from threading import Lock

if __name__ == "__main__":
    import sys
    sys.path.append("../")

try:
    from base import BaseDataProducer
except:
    from processers.base import BaseDataProducer
from utils.interpret import task2_interpretor, task3_interpretor

import os
import pandas as pd
import sqlite3
import random
import numpy as np
import time
INSUFFICIENT_OSM_DATA_CODE = 10


class DefaultDataProducer(BaseDataProducer):

    _SQL_QUERY_METAS = """
                select p.IMAGE_NAME, p.NAME, p.SAMPLE_CENTER_X, p.SAMPLE_CENTER_Y, p.DIMENSION, p.DOWNLOAD_TIME, p.CRS, i.SCALE, p.ID, p.STATUS, p.NUM_MAP_ELEMENTS, p.NUM_ANNOTATIONS from patch p
                left join image i on p.IMAGE_NAME = i.NAME where p.NUM_ANNOTATIONS is NULL;
                """
    _SQL_QUERY_OSM = """
                   select OSM_ID, TAGS, GEOMETRY from osm_data where PATCH_NAME =? and TYPE in ({});   
                   """

    _SQL_QUERY_OSM_STATISTICS = """
                                SELECT PATCH_NAME, COUNT(CASE WHEN TYPE = 0 THEN 1 END) as BG_CNT, 
                                                COUNT(CASE WHEN TYPE = 1 THEN 1 END) as AREA_CNT, 
                                                COUNT(CASE WHEN TYPE = 2 THEN 1 END) as NONE_AREA_CNT 
                                                FROM osm_data 
                                                WHERE PATCH_NAME = ? 
                                                GROUP BY PATCH_NAME;
                    """

    def __init__(
        self,
        queue,
        lock: Lock,
        affair_queue: PriorityQueue,
        affair_lock,
        db_root,
        metadata_db,
        osm_db,
        annotation_db,
        osm_wiki_db,
        policy_config: dict,
        producer_id=None,
        prefetch_size=100,
        map_element_threshold=10,
        tick_interval=0.01,
    ):
        """_summary_

        Args:
            queue (queue.Queue): Used to store the results of the data producer.
            lock (Lock): lock to protect the queue.
            affair_queue (queue.Queue): Use to store the signals for the scheduler.
            affair_lock (Lock): lock to protect the affair_queue.
            db_root (str): The root path of the databases. if all the databases are in the same root, then it is the same as the root path. Otherwise, it is the parent directory of the databases.
            metadata_db (str): _description_
            osm_db (str): _description_
            annotation_db (str): _description_
            policy_config (dict): a dictionary containing the policy configuration.
            prefetch_size (int, optional): The number of meta data to fetch each time. Defaults to 100.
            map_element_threshold (int, optional): Only fetch metadata with more than this number of map elements. Defaults to 10.
        """
        super(DefaultDataProducer, self).__init__(
            queue, lock, affair_queue, affair_lock, producer_id, tick_interval
        )

        self.db_root = db_root if os.path.isdir(db_root) else "/"
        self.metadata_db = (
            metadata_db
            if metadata_db.startswith("/")
            else os.path.join(db_root, metadata_db)
        )
        self.osm_db = (
            osm_db if osm_db.startswith("/") else os.path.join(db_root, osm_db)
        )
        self.annotation_db = (
            annotation_db
            if annotation_db.startswith("/")
            else os.path.join(db_root, annotation_db)
        )
        self.osm_wiki_db = (
            osm_wiki_db
            if osm_wiki_db.startswith("/")
            else os.path.join(db_root, osm_wiki_db)
        )

        self.policy_config = policy_config

        self.prefetch_size = prefetch_size
        self.map_element_threshold = map_element_threshold

        self.taskx_interpreter = {}
        self.rng = random.Random()
        # set the seed for the random number generator
        self.rng.seed(int(time.time()) + self.producer_id)
        self.init_interpreters(self.policy_config)
        
    def init_database(self):

        # set the seed for the numpy random number generator
        np.random.seed(int(time.time()) + self.producer_id)
        # set the seed for random module
        random.seed(self.producer_id)

        self.conn_metadata = sqlite3.connect(self.metadata_db, timeout=300)
        self.conn_osm = sqlite3.connect(self.osm_db, timeout=300)
        self.conn_annotation = sqlite3.connect(self.annotation_db, timeout=300)
        # caching 40G osm data usually takes 3 minutes
        self.osm_table = pd.read_sql_query(
            "SELECT * FROM osm_data WHERE TYPE = 1",
            self.conn_osm,
        )
        self.logger.info("Connected to the database")

    def deinit_database(self):

        self.logger.info("Disconnecting from the database")
        self.conn_metadata.close()
        self.conn_osm.close()
        self.conn_annotation.close()
        self.logger.info("Disconnected from the database")

    def init_interpreters(self, policy_config):
        """
        Initialize the interpreters according to the policy configuration.
        """
        # TODO: implement the policy configuration dynamically
        # for now, we use the default policy configuration
        self.taskx_interpreter = {2: task2_interpretor, 3: task3_interpretor}

    def prefetch_data(self):
        """
        Cache necessary metadata from the database. Select a batch of images. Return a list of patch ids.
        """
        
        # release the memory of the previous prefetch data
        self.metadata_table = None
        self.annotator_prompt_table = None
        self.prompt_template_table = None
        
        self.metadata_table = pd.read_sql_query(
            self._SQL_QUERY_METAS, self.conn_metadata
        )

        # cache annotation metadata
        self.annotator_prompt_table = pd.read_sql_query(
            "SELECT * FROM annotator_prompt",
            self.conn_annotation,
        )
        
        self.prompt_template_table = pd.read_sql_query(
            "SELECT * FROM prompt",
            self.conn_annotation,
        )
        
        # filter the metadata table by map element threshold
        self.metadata_table = self.metadata_table.loc[
            self.metadata_table["NUM_MAP_ELEMENTS"] > self.map_element_threshold
        ]

        # filter the metadata table by status,
        # statue ！= 0 means it is marked as an invalid patch, skip it
        self.metadata_table = self.metadata_table.loc[
            self.metadata_table["STATUS"] != 0
        ]

        # sample the metadata table with 10 x prefetch_size
        self.metadata_table = self.metadata_table.sample(frac=1).iloc[: self.prefetch_size * 10]

        # set the NaN values in NUM_ANNOTATIONS to 0
        self.metadata_table["NUM_ANNOTATIONS"] = self.metadata_table.loc[
            "NUM_ANNOTATIONS"
        ].fillna(0)
        
        # sort the metadata table by number of annotations from low to high
        self.metadata_table = self.metadata_table.sort_values(
            by=["NUM_ANNOTATIONS"], ascending=True
        )

        # reset the index of the metadata table
        self.metadata_table.reset_index(drop=True, inplace=True)

        # get the first prefetch_size meta data
        self.metadata_table = self.metadata_table.iloc[: self.prefetch_size]

        sampled_patches = self.metadata_table["ID"].tolist()

        return sampled_patches

    def task_selector(self, image_name, patch_name, patch_id):
        """Select a task type and osm data type for the given patch according to the policy configuration.
        TODO: implement the policy configuration dynamically. For now, we use the default policy configuration.
        Args:
            image_name (str): The name of the image where the patch is croped from.
            patch_name (str): The name of the patch.
            patch_id (int): The id of the patch in the patch table.

        Returns:
            tuple|None: A tuple containing the task type and osm data type.
        """

        task_osm_lut = {2: [1], 3: [2]}

        # get the image name and patch name from the metadata table
        meta_data = self.metadata_table.loc[self.metadata_table["ID"] == patch_id]
        if len(meta_data) < 1:
            # if there is no metadata, skip this iteration
            self.logger.error(
                f"No metadata for patch id {patch_id}, skip this iteration."
            )
            return None

        # query the osm statistics from the osm table
        # legacy code, we used sqlite to query the osm statistics, but it is too slow for large datasets
        # c_osm_stat = self.conn_osm.cursor()
        # c_osm_stat.execute(
        #     self._SQL_QUERY_OSM_STATISTICS, ("/".join([image_name, patch_name]),)
        # )
        # osm_stat = c_osm_stat.fetchone()

        osm_elements = self.osm_table.loc[
            (self.osm_table["PATCH_NAME"] == "/".join([image_name, patch_name]))
        ]
        osm_stat = osm_elements.groupby('TYPE')[['ID']].count().reset_index().rename(columns={'ID': 'COUNT'})

        if osm_stat['COUNT'].sum() < 1:
            # if there is no osm data, skip this task
            self.logger.error(
                f"No osm data for {image_name}/{patch_name}, skip this task. Patch id: {patch_id}."
            )
            return None

        # ramdomly select a task within task type 2 and task type 3
        # task type 2: area decription, requires osm data type 1, which is reflated by area_cnt
        # task type 3: none-area decription, requires osm data type 2, which is reflated by none_area_cnt
        # not all patches have both area and none-area osm data, so we need to check the osm data type before selecting the task type
        # choose the corresponding task type if only one type of osm data is available
        # skip this task if there are only background osm data
        available_task_types = []
        if osm_stat[osm_stat['TYPE'] == 1]['COUNT'].sum() > 0:
            available_task_types.append(2)
        if osm_stat[osm_stat['TYPE'] == 2]['COUNT'].sum() > 0:
            available_task_types.append(3)

        if len(available_task_types) < 1:
            # if there is no osm data, skip this task
            self.logger.error(
                f"No suitable task type for {image_name}/{patch_name}, skip this patch. Patch id: {patch_id}."
            )
            # update the status of the patch in the metadata table
            # FIXME: this is a hack, we should use a more elegant way to update the status of the patch
            self.metadata_table.loc[
                self.metadata_table["ID"] == patch_id, "STATUS"
            ] += INSUFFICIENT_OSM_DATA_CODE
            # update the sqlite database
            c = self.conn_metadata.cursor()
            c.execute(
                "UPDATE patch SET STATUS =? WHERE ID =?",
                (
                    self.metadata_table.loc[
                        self.metadata_table["ID"] == patch_id, "STATUS"
                    ].values[0],
                    patch_id,
                ),
            )
            self.conn_metadata.commit()
            return None
        
        if len(available_task_types) == 2:
            # FIXME: this is a hack. We find that non-area is more in quantity than area, so we give a higher weight to area.
            available_task_types.sort()
            task_type = self.rng.choices(available_task_types, weights=[0.6, 0.4], k=1)[0]
        else:
            task_type = self.rng.choice(available_task_types)

        # choose the corresponding osm data type
        osm_data_type = task_osm_lut[task_type]

        return task_type, osm_data_type, osm_elements

    def prompt_selector(self, task_type):
        """
        Select a prompt template according to the task type. Also fetch the valid annotator ids for this prompt.
        TODO: implement the prompt template dynamically. For now, we use the default prompt template.
        Args:
            task_type (int): The task type.

        Returns:
            tuple|None: A tuple containing the prompt id, prompt template, and valid annotator ids.
        """
        # fetch a prompt template according to the task type
        prompt_templates = self.prompt_template_table[
            self.prompt_template_table["TYPE"] == task_type
        ][["ID", "PROMPT"]]
        if len(prompt_templates) < 1:
            # if there is no prompt template, skip this task
            self.logger.error(
                f"No prompt template for task {task_type}, skip this task"
            )
            return None
        seleted_prompt_template = prompt_templates.sample(n=1).iloc[0]
        prompt_id, prompt_template = seleted_prompt_template['ID'], seleted_prompt_template['PROMPT']
        # fetch the valid annotator ids for this prompt
        annotator_ids = self.annotator_prompt_table[
            self.annotator_prompt_table["PROMPT"] == prompt_id
        ]["ANNOTATOR"].tolist()
        
        if len(annotator_ids) < 1:
            # if there is no valid annotator, skip this task
            self.logger.error(
                f"No valid annotator for prompt {prompt_id}, skip this task"
            )
            return None

        return prompt_id, prompt_template, annotator_ids

    def generate_task(self, data):

        patch_id = data
        meta_data = self.metadata_table.loc[self.metadata_table["ID"] == patch_id]
        image_name, patch_name = (
            meta_data["IMAGE_NAME"].values[0],
            meta_data["NAME"].values[0],
        )

        task_selecter_result = self.task_selector(image_name, patch_name, patch_id)
        if task_selecter_result is None:
            # if there is no task, skip this iteration
            return None

        task_type, osm_data_type, osm_elements = task_selecter_result

        prompt_selector_result = self.prompt_selector(task_type)
        if prompt_selector_result is None:
            # if there is no prompt, skip this task
            return None

        prompt_id, prompt_template, annotator_ids = prompt_selector_result

        osm_elements = osm_elements[osm_elements['TYPE'].isin(osm_data_type)]
        
        self.logger.info(
            f"Fetched {len(osm_elements)} osm elements for {image_name}/{patch_name}. Patch id: {patch_id}."
        )

        if len(osm_elements) < 1:
            # if there is no osm data, skip this task
            self.logger.error(
                f"No osm data for {image_name}/{patch_name}, skip this task. Patch id: {patch_id}."
            )
            return None
        # center_x, center_y, dimension, download_time, patch_crs, scale are needed as a tuple
        interpretation_meta_data = (
            None,
            None,
            meta_data["SAMPLE_CENTER_X"].values[0],
            meta_data["SAMPLE_CENTER_Y"].values[0],
            meta_data["DIMENSION"].values[0],
            meta_data["DOWNLOAD_TIME"].values[0],
            meta_data["CRS"].values[0],
            meta_data["SCALE"].values[0],
        )
        try:
            interpretation = self.taskx_interpreter[task_type](
                interpretation_meta_data,
                osm_elements,
                self.osm_wiki_db,
                prompt_template,
                self.policy_config,
            )
        except Exception as e:
            self.logger.error(f"Error in interpreting the task. Patch id: {patch_id}. Task type: {task_type}, OSM data type: {osm_data_type}. Error: {e}")
            return None

        if interpretation is None:
            # if interpretation is None, it means the task is not suitable for this patch, skip this task
            self.logger.error(
                f"No suitable interpretation for patch id: {patch_id}. Task type: {task_type}, OSM data type: {osm_data_type}"
            )
            c = self.conn_metadata.cursor()
            c.execute(
                "UPDATE patch SET STATUS =? WHERE ID =?",
                (
                    20,
                    patch_id,
                ),
            )
            self.conn_metadata.commit()
            
            return None

        # FIXME: use patch_id as the key may not be unique. As there may be duplicate downloads of the same patch, the key may be the same.
        task = {
            "type": "annotation_task",
            "patch_id": int(patch_id),
            "prompt_id": int(prompt_id),
            "valid_annotator_ids": annotator_ids,
        }

        task.update(interpretation)

        return task

# test the data producer
if __name__ == "__main__":

    import ee
    import json
    ee.Initialize()

    queue = Queue()
    lock = Lock()
    affair_queue = PriorityQueue()
    affair_lock = Lock()
    db_root =  '/mnt/FastDisk/GeJunYao/VLP/databases'
    metadata_db = "metadata.db"
    osm_db = "osm.db"
    annotation_db = "annotation.db"
    osm_wiki_db = "taginfo-wiki.db"
    policy_config = json.load(open("/mnt/SrvUserDisk/Gejunyao/VLP/RSVLD/annotation/configs/policy_config.json", "r"))
    prefetch_size = 10
    map_element_threshold = 5
    data_producer = DefaultDataProducer(
        queue,
        lock,
        affair_queue,
        affair_lock,
        db_root,
        metadata_db,
        osm_db,
        annotation_db,
        osm_wiki_db,
        policy_config,
        prefetch_size=prefetch_size,
        map_element_threshold=map_element_threshold,
    )
    
    data_producer.start()
    try:
        while True:
            time.sleep(1)
            if not queue.empty():
                task = queue.get()
                print(task)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        data_producer.stop()
        data_producer.join()
        print("Data producer stopped")