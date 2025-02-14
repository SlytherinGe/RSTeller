"""
This is the main script for annotating images with Google Earth Engine and uploading the results to a database.
There two processes:
1. Data producer: This process queries the osm data in local database and selects the image patches to be annotated. It then queries current running LLM backends in local server and turn the raw osm data into prompts according to available backends. Finally, it sends the prompts to the LLM backends.
2. Data collector: This process receives the annotated results from the LLM backends and uploads the results to the local database.
"""

import os
import sqlite3
import time
import random
from multiprocessing import Process, Event
from queue import PriorityQueue, Queue
from multiprocessing import Queue as MPQueue
from threading import Thread, Lock
import json

import logging
from logging.handlers import RotatingFileHandler

import ee

from utils.interpret import (
    annotation_parser_prompt1_annotator1,
    annotation_parser_prompt4_annotator,
)

from annotator_connector import AnnotatorConnector
from processers.rewrite_data_producer import RewriteDataProducer
from processers.default_data_producer import DefaultDataProducer

import numpy as np
import argparse



class DataSaver(Process):
    """This saver saves the annotated results to the annotation database."""

    def __init__(self, queue, db_root, metadata_db, annotation_db, tick_interval=0.01):
        """Initialize the data saver.
            There are two tables in the annotation database:
            1. annotation: store the annotated results. It has the following columns:
                - ID: the id of the annotation. It is the primary key. It is auto-incremented.
                - PATCH: the id of the patch.
                - ANNOTATION: the annotation.
                - NUM_ELEMS: the number of map elements used in the annotation.
                - ANNOTATOR: the id of the annotator.
                - PROMPT: the id of the prompt.
            2. annotation_osm: store the mapping between annotation and osm data. It has the following columns:
                - ID: the id of the mapping. It is the primary key. It is auto-incremented.
                - ANNOTATION: the id of the annotation.
                - OSM_ID: the id of the osm data.
        Args:
            queue (queue.Queue): The queue to receive the annotated results.
                                       The format of the task is:
                                       {'type': 'save_annotation',
                                        'patch_id': int,
                                        'annotator_id': int,
                                        'prompt_id': int,
                                        'osm_ids': list(int),
                                        'annotation': str}
            db_root (str): The root path of the databases.
            metadata_db (str): The path to the metadata database.
            annotation_db (str): The path to the annotation database.
            tick_interval (float, optional): The time interval to check the queue. Defaults to 0.01.
        """
        super(DataSaver, self).__init__()
        self.queue = queue
        self.db_root = db_root
        self.metadata_db = (
            metadata_db
            if metadata_db.startswith("/")
            else os.path.join(db_root, metadata_db)
        )
        self.annotation_db = (
            annotation_db
            if annotation_db.startswith("/")
            else os.path.join(db_root, annotation_db)
        )
        self.tick_interval = tick_interval

        self.logger = logging.getLogger("[DataSaver]")

        self.exit_flag = Event()
        self.exit_flag.clear()

    def run(self):
        # connect to the database
        self.logger.info("Connecting to the database")

        self.conn_annotation = sqlite3.connect(self.annotation_db, timeout=360)
        self.conn_metadata = sqlite3.connect(self.metadata_db, timeout=360)
        self.logger.info("Connected to the database")

        while not self.exit_flag.is_set():
            if self.queue.empty():
                time.sleep(self.tick_interval)
                continue

            task = self.queue.get()
            self.logger.info(f'Received task {task["type"]}')

            if task["type"] == "annotation_response":
                patch_id, annotation, osm_ids, annotator_id, prompt_id = (
                    task["patch_id"],
                    task["annotation"],
                    task["osm_ids"],
                    task["annotator_id"],
                    task["prompt_id"],
                )

                # TODO implement the annotation parser dynamically
                # for now, we use the default annotation parser
                try:
                    # if prompt_id == 4:
                    #     annotation = annotation_parser_prompt4_annotator(annotation)
                    # else:
                    #     annotation = annotation_parser_prompt1_annotator1(annotation)
                    annotation = annotation.strip()
                except Exception as e:
                    self.logger.error(
                        f"Error in parsing the annotation:\n {annotation} \n {e}"
                    )
                    continue

                # save the annotation to the database
                c = self.conn_annotation.cursor()
                c.execute(
                    "INSERT INTO annotation (PATCH, ANNOTATION, NUM_ELEMS, ANNOTATOR, PROMPT) VALUES (?,?,?,?,?)",
                    (patch_id, annotation, len(osm_ids), annotator_id, prompt_id),
                )
                annotation_id = c.lastrowid

                for osm_id in task["osm_ids"]:
                    c.execute(
                        "INSERT INTO annotation_osm (ANNOTATION, OSM_ID) VALUES (?,?)",
                        (annotation_id, osm_id),
                    )

                # add a annotation count to the patch table, and set to 1 if it is null
                c = self.conn_metadata.cursor()
                c.execute(
                    "UPDATE patch SET NUM_ANNOTATIONS = (COALESCE(NUM_ANNOTATIONS, 0) + 1) WHERE ID = ?",
                    (task["patch_id"],),
                )
                self.conn_annotation.commit()
                self.conn_metadata.commit()
                self.logger.info(
                    f'Saved annotation {annotation_id} for patch {task["patch_id"]}, prompt_id {prompt_id}, annotator_id {annotator_id}'
                )

        self.disconnect_from_db()

    def disconnect_from_db(self) -> None:
        self.logger.info("Disconnecting from the database")
        self.conn_annotation.close()
        self.logger.info("Disconnected from the database")

    def stop(self) -> None:
        self.logger.info("Closing the data saver")
        self.exit_flag.set()
        self.logger.info("Closed the data saver")


class ManagerProcess(Thread):
    """This is the manager process that runs the data producer, data worker.
    It hosts the connection to the annotator backend and the scheduler.
    It receives the commands from the CLI and executes them.
    Users can use this interface to start or stop the connections to the annotator backend.
    (The connection is achieved by class AnnotatorConnector, which is a wrapper of socket.)
    """

    def __init__(
        self,
        db_root,
        metadata_db,
        osm_db,
        annotation_db,
        osm_wiki_db,
        policy_config,
        num_data_producer=4,
        prefetch_size=100,
        map_element_threshold=10,
        tick_interval=0.01,
    ):
        """Initialize the main process.
            There are two sub processes:
            1. Data producer: This process queries the osm data in local database and selects the image patches to be annotated.
               It then queries current available prompt templates in the database and generates the tasks.
               Then it puts the tasks into the queue to be consumed by main process.
            2. Data saver: This process receives the annotated results in the queue and saves them to the annotation database.
        Args:
            db_root (str): The root path to the database.
            metadata_db (str): The name of the metadata database.
            osm_db (str): The name of the osm database.
            annotation_db (str): The name of the annotation database.
            osm_wiki_db (str): The name of the osm wiki database.
            policy_config (dict): The policy configuration.
            prefetch_size (int, optional): The number of metadata to prefetch. Defaults to 100.
            map_element_threshold (int, optional): The threshold of the number of map elements. Defaults to 10.
            tick_interval (float, optional): The time interval to check the queue. Defaults to 0.01.
        """
        super(ManagerProcess, self).__init__()
        self.db_root = db_root
        self.metadata_db = metadata_db
        self.osm_db = osm_db
        self.annotation_db = annotation_db
        self.osm_wiki_db = osm_wiki_db
        self.policy_config = policy_config
        self.prefetch_size = prefetch_size
        self.map_element_threshold = map_element_threshold
        self.tick_interval = tick_interval
        self.num_data_producer = num_data_producer

        self.logger = logging.getLogger("[MainProcess]")

        self.exit_flag = Event()
        self.exit_flag.clear()

        self.task_queue = MPQueue(maxsize=1000)
        self.task_lock = Lock()

        self.affair_queue = PriorityQueue(maxsize=1000)
        self.affair_lock = Lock()

        self.save_queue = MPQueue(maxsize=1000)
        self.save_lock = Lock()

        # This is a dictionary to store the annotator connections.
        # The key is the annotator id and the value is a list of sub-dicts, each of which contains the following keys:
        # - dest_addr: the destination address of the annotator in the format of ip:port.
        # - last_activity (optional): the last activity time of the annotator.
        # - queue: the queue to send the tasks to the annotator.
        # - lock: the lock to synchronize the access to the queue.
        # - connector: the instance of the AnnotatorConnector to communicate with the annotator.
        self.annotator_connections = {}

        self.data_producers = [
            DefaultDataProducer(
                self.task_queue,
                self.task_lock,
                self.affair_queue,
                self.affair_lock,
                self.db_root,
                self.metadata_db,
                self.osm_db,
                self.annotation_db,
                self.osm_wiki_db,
                self.policy_config,
                i,
                self.prefetch_size,
                self.map_element_threshold,
                self.tick_interval,
            )
            for i in range(self.num_data_producer)
        ]

        rewrite_producer = RewriteDataProducer(
            self.task_queue,
            self.task_lock,
            self.affair_queue,
            self.affair_lock,
            self.db_root,
            self.annotation_db,
            "annotation_meta.db",
            max_rewrites=3,
            prefetch_size=10000,
            producer_id=len(self.data_producers) + 1,
        )
        self.data_producers.append(rewrite_producer)

        self.data_worker = DataSaver(
            self.save_queue,
            self.db_root,
            self.metadata_db,
            self.annotation_db,
            self.tick_interval,
        )

        # TODO: use the policy config or the arguments to set up the annotator connections dynamically
        # for now, we use the following hard-coded connections
        self.add_annotator("localhost", 5000)
        self.add_annotator("localhost", 5001)
        self.add_annotator("localhost", 5002)
        self.add_annotator("localhost", 5003)

    def add_annotator(self, host, port):
        # set up an annotator connector to communicate with the annotator backend for testing

        if host == "localhost":
            host = "127.0.0.1"

        # TODO: check if the annotator is already connected
        # if the annotator is already connected, return the annotator id

        # create a new annotator connector
        annotator_queue = Queue(maxsize=10)
        annotator_lock = Lock()
        annotator_connector = AnnotatorConnector(
            host,
            port,
            annotator_queue,
            annotator_lock,
            self.save_queue,
            self.save_lock,
            self.affair_queue,
            self.affair_lock,
            self.tick_interval,
        )
        annotator_id = annotator_connector.get_annotator_id()

        if annotator_id not in self.annotator_connections:
            self.annotator_connections[annotator_id] = []

        self.annotator_connections[annotator_id].append(
            {
                "dest_addr": f"{host}:{port}",
                "queue": annotator_queue,
                "lock": annotator_lock,
                "connector": annotator_connector,
            }
        )
        # start the annotator connector
        annotator_connector.start()
        self.logger.info("Started the annotator connector")

        return annotator_id

    def run(self):
        # start the data producer
        self.logger.info("Starting the data producer")
        for data_producer in self.data_producers:
            data_producer.start()
        self.logger.info("Started the data producer")

        # start the data worker
        self.logger.info("Starting the data worker")
        self.data_worker.start()
        self.logger.info("Started the data worker")

        while not self.exit_flag.is_set():

            # check the task queue and distribute the tasks to the annotator connections
            if not self.task_queue.empty():
                task = self.task_queue.get()
                self.logger.debug(f'Received task {task["type"] }')
                if task["type"] == "annotation_task":
                    # distribute the task to the annotator connections
                    valid_annotator_ids = []
                    for annotator_id in task["valid_annotator_ids"]:
                        if annotator_id in self.annotator_connections:
                            valid_annotator_ids.append(annotator_id)

                    # disgard the task if there is no valid annotator connection
                    if len(valid_annotator_ids) == 0:
                        self.logger.info(
                            f"No valid annotator connection for task {task}"
                        )
                        continue

                    # distribute the task to the valid annotator connections
                    # sort all the valid annotator connections by the length of their queue
                    for valid_annotator_id in valid_annotator_ids:
                        self.annotator_connections[valid_annotator_id] = sorted(
                            self.annotator_connections[valid_annotator_id],
                            key=lambda x: x["queue"].qsize(),
                        )

                    sorted_annotator_ids = sorted(
                        valid_annotator_ids,
                        key=lambda x: self.annotator_connections[x][0]["queue"].qsize(),
                    )

                    task["type"] = "annotation_request"
                    self.logger.info(
                        f'Distributing task with prompt_id {task["prompt_id"]} to annotators {sorted_annotator_ids}'
                    )
                    # aquire the lock to send the task to the annotator
                    self.annotator_connections[sorted_annotator_ids[0]][0]["queue"].put(
                        task
                    )
                    self.logger.info(
                        f'Sent task to annotator {self.annotator_connections[sorted_annotator_ids[0]][0]["dest_addr"]}'
                    )

            # wait for a while
            time.sleep(self.tick_interval)

        # stop the data producer
        self.logger.info("Stopping the data producer")
        for data_producer in self.data_producers:
            data_producer.stop()
        self.logger.info("Stopped the data producer")

        # stop the data worker
        self.logger.info("Stopping the data worker")
        self.data_worker.stop()
        self.logger.info("Stopped the data worker")

        # wait for the data producer to exit
        for data_producer in self.data_producers:
            data_producer.join()
        self.logger.info("Data producer exited")

        # wait for the data worker to exit
        self.data_worker.join()
        self.logger.info("Data worker exited")

    def stop(self) -> None:
        self.logger.info("Closing the main process")
        self.exit_flag.set()
        self.logger.info("Closed the main process")


if __name__ == "__main__":

    # ignore np divide by zero warning
    np.seterr(divide="ignore", invalid="ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_root",
        type=str,
        default="../database",
        help="Path to the data file",
    )
    parser.add_argument(
        "--metadata_db",
        type=str,
        default="metadata.db",
        help="the metadata sqilite database file name",
    )
    parser.add_argument(
        "--osm_db", type=str, default="osm.db", help="the osm sqilite database name"
    )
    parser.add_argument(
        "--annotation_db",
        type=str,
        default="annotation.db",
        help="the annotation sqilite database name",
    )
    parser.add_argument(
        "--osm_wiki_db",
        type=str,
        default="taginfo-wiki.db",
        help="the osm wiki sqilite database path",
    )
    parser.add_argument(
        "--log_file", type=str, default="logs/annotation.log", help="the log file name"
    )
    parser.add_argument(
        "--fetch_size",
        type=int,
        default=100,
        help="the number of data to fetch each time",
    )
    parser.add_argument(
        "--tick_interval",
        type=float,
        default=0.01,
        help="the time interval to check the queue",
    )
    parser.add_argument(
        "--map_element_threshold",
        type=int,
        default=10,
        help="the threshold of the number of map elements",
    )
    parser.add_argument(
        "--num_data_producer",
        type=int,
        default=4,
        help="the number of data producer processes",
    )
    parser.add_argument(
        "--service_account",
        type=str,
        default=None,
        help="The service account to use for GEE authentication.",
    )
    parser.add_argument(
        "--credentials_file",
        type=str,
        default=None,
        help="The file to save the GEE credentials.",
    )
    parser.add_argument(
        "--policy_config",
        type=str,
        default='configs/policy_config.json',
        help="The policy configuration file, in JSON format.",
    )

    args = parser.parse_args()

    credentials = ee.ServiceAccountCredentials(args.service_account, args.credentials_file)
    ee.Initialize(credentials)
    # test manager process
    db_root = args.db_root
    metadata_db = args.metadata_db
    osm_db = args.osm_db
    annotation_db = args.annotation_db
    osm_wiki_db = args.osm_wiki_db
    log_file = args.log_file
    prefetch_size = args.fetch_size
    map_element_threshold = args.map_element_threshold
    tick_interval = args.tick_interval
    num_data_producer = args.num_data_producer

    policy_config = json.load(open(args.policy_config))

    log_format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s"
    logging.basicConfig(
        handlers=[RotatingFileHandler(log_file, maxBytes=10000000, backupCount=9)],
        level=logging.INFO,
        format=log_format,
    )

    manager_process = ManagerProcess(
        db_root,
        metadata_db,
        osm_db,
        annotation_db,
        osm_wiki_db,
        policy_config,
        num_data_producer=num_data_producer,
        prefetch_size=prefetch_size,
        map_element_threshold=map_element_threshold,
        tick_interval=tick_interval,
    )
    manager_process.start()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            manager_process.stop()
            break
    manager_process.join()
