from multiprocessing import Queue
from queue import PriorityQueue
from threading import Lock
try:
    from base import BaseDataProducer
except:
    from processers.base import BaseDataProducer
import os
import pandas as pd
import sqlite3
import random

class RewriteDataProducer(BaseDataProducer):
    
    def __init__(self, queue: Queue, 
                       lock: Lock, 
                       affair_queue: 
                       PriorityQueue, 
                       affair_lock: Lock, 
                       db_root,
                       annotation_db,
                       annotation_meta_db,
                       max_rewrites=5, 
                       prefetch_size=100, 
                       producer_id=None, 
                       tick_interval=0.01):
        super().__init__(queue, lock, affair_queue, affair_lock, producer_id, tick_interval)
        
        self.db_root = db_root
        self.annotation_db = annotation_db if annotation_db.startswith('/') \
                             else os.path.join(db_root, annotation_db)
        self.annotation_meta_db = annotation_meta_db if annotation_meta_db.startswith('/') \
                             else os.path.join(db_root, annotation_meta_db)
        self.max_rewrites = max_rewrites
        self.prefetch_size = prefetch_size
        
                             
    def init_database(self):
        
        self.conn_anno = sqlite3.connect(self.annotation_db)
        self.conn_anno_meta = sqlite3.connect(self.annotation_meta_db)
        
    def deinit_database(self):
        
        self.conn_anno.close()
        self.conn_anno_meta.close()

    def prefetch_data(self):
        
        # load table annotation, annotator_prompt, annotation_osm into memory
        c = self.conn_anno.cursor()
        c.execute("SELECT * FROM annotation")
        self.annotation_table = pd.DataFrame(c.fetchall(), columns=[desc[0] for desc in c.description])
        c.execute("SELECT * FROM annotator_prompt")
        self.annotator_prompt_table = pd.DataFrame(c.fetchall(), columns=[desc[0] for desc in c.description])
        c.execute("SELECT * FROM annotation_osm")
        self.annotation_osm_table = pd.DataFrame(c.fetchall(), columns=[desc[0] for desc in c.description])
        
        # load prompt template for rewrite into memory
        c.execute("select * from prompt where type = 11")
        self.prompt_template = pd.DataFrame(c.fetchall(), columns=[desc[0] for desc in c.description])
        
        # load rewrite examples into memory
        c_meta = self.conn_anno_meta.cursor()
        c_meta.execute('select * from rewrite_raw')
        self.rewrite_raw_table = pd.DataFrame(c_meta.fetchall(), columns=[desc[0] for desc in c_meta.description])
        c_meta.execute('select * from rewrite_examples')
        self.rewrite_examples_table = pd.DataFrame(c_meta.fetchall(), columns=[desc[0] for desc in c_meta.description])
        
        # exclude the broken patch ids
        self.annotation_table['patch_valid'] = self.annotation_table['PATCH'].apply(lambda x: isinstance(x, int))
        self.annotation_table = self.annotation_table[self.annotation_table['patch_valid'] == True]
        
        # count the number of duplicate annotations
        duplicate_count = self.annotation_table.groupby(['PATCH']).size().reset_index(name='count')
        duplicate_count.sort_values(by='count', ascending=True, inplace=True)
        
        # keep the patches with less than max_rewrites annotations
        valid_patches = duplicate_count[duplicate_count['count'] < self.max_rewrites]['PATCH']
        
        # sample prefetch_size patches randomly
        # sampled_patches = np.random.choice(valid_patches, self.prefetch_size, replace=False).tolist()
        sampled_patches = valid_patches[:self.prefetch_size].tolist()
        
        self.logger.info(f'Prefetching {len(sampled_patches)} patches.')
        
        return sampled_patches
    
    def generate_task(self, data):
        
        """
        Rewrite task generator.
        
        Args:
            data (int): patch id that needs to be rewritten.
        
        Returns: 
            task (dict): a dictionary containing the patch id and the corresponding rewrite candidates.
        """
        # find the corresponding annotation according to the patch id
        annotation_row = self.annotation_table[self.annotation_table['PATCH'] == data]
        annotation_row.loc[:, 'CREATED_AT'] = pd.to_datetime(annotation_row['CREATED_AT'])
        # find the oldest annotation as the base for the rewrite
        # try:
        oldest_annotation = annotation_row.loc[annotation_row['CREATED_AT'] ==annotation_row['CREATED_AT'].min()]
        oldest_annotation = oldest_annotation.iloc[0]
        # except Exception as e:
        #     self.logger.error(f'Error in finding the oldest annotation for patch {data}: {e}')
        #     return None
        # find the corresponding annotator prompt
        ori_prompt_id = oldest_annotation['PROMPT']
        # find the rewrites
        rewrite_examples = self.rewrite_raw_table[self.rewrite_raw_table['prompt_id']==ori_prompt_id].\
                            merge(self.rewrite_examples_table, left_on='id', right_on='rewrite_raw_id', how='left')
        selected_examples = []
        for _, table in rewrite_examples.groupby('rewrite_raw_id'):
            row = table.sample(n=1).iloc[0]
            selected_examples.append(row)

        # shuffle the examples
        random.shuffle(selected_examples)
        # sample a prompt template
        prompt_id, template = self.prompt_template.sample(n=1).iloc[0][['ID', 'PROMPT']]
        # combine the selected examples and the prompt template
        format_kwargs = dict()
        for i, example in enumerate(selected_examples):
            example_id = f'example{i+1}'
            format_kwargs[example_id+'_raw'] = example['text']
            format_kwargs[example_id+'_revised'] = example['rewrite_text']
        format_kwargs['raw'] = oldest_annotation['ANNOTATION']

        formated_prompt = template.format(**format_kwargs)   
        
        # get the osm ids related to the given annotation id
        anno_id = oldest_annotation['ID']
        osm_ids = self.annotation_osm_table[self.annotation_osm_table['ANNOTATION'] == anno_id]['OSM_ID'].tolist()  
        # select valid annotator ids
        valid_annotators = self.annotator_prompt_table[self.annotator_prompt_table['PROMPT']==prompt_id]['ANNOTATOR'].tolist()
        # build the task
        task = dict(type='annotation_task',
                    prompt=formated_prompt,
                    osm_ids=osm_ids,
                    patch_id=int(oldest_annotation['PATCH']),
                    prompt_id=int(prompt_id),
                    valid_annotator_ids=valid_annotators)
        
        return task
    
    
    
# test the RewriteDataProducer
if __name__ == '__main__':
    import time
    
    db_root = '/mnt/FastDisk/GeJunYao/VLP/databases'
    annotation_db = 'annotation.db'
    annotation_meta_db = 'annotation_meta.db'
    max_rewrites = 5
    prefetch_size = 100
    producer_id = 0
    tick_interval = 0.01
    
    queue = Queue()
    lock = Lock()
    affair_queue = PriorityQueue()
    affair_lock = Lock()
    print('Starting data producer...')
    data_producer = RewriteDataProducer(queue, lock, affair_queue, affair_lock, db_root, annotation_db, annotation_meta_db, max_rewrites, prefetch_size, producer_id, tick_interval)
    data_producer.start()
    print('Data producer started.')
    try:
        while True:
            time.sleep(1)
            if not queue.empty():
                task = queue.get()
                print(task)
    except KeyboardInterrupt:
        print('KeyboardInterrupt received, stopping data producer...')
        data_producer.stop()
        data_producer.join()
        print('Data producer stopped.')