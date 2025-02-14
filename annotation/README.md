# A Distributed Annotation System for RSTeller

## System Overview

![System Overview](../assets/annotation_system.jpg)

The system consists of two main components: the annotator frontend and the annotator backend. The annotator backend hosts the LLM captioning service and each instance of the backend runs an LLM service. The backend can be easily swapped with any other LLM implementation, whether it runs on a GPU, CPU, or is accessed via an API. The annotator frontend is a multi-process application that manages three process groups: producer process, saver process, and connector process.

- **Producer Process**: Responsible for reading raw OSM data and transforming it into formatted prompts for the annotator backend.
- **Saver Process**: Handles parsing the responses from the annotator backend and saving them to the database. This design allows for easy database swapping, as well as adding new data producers or parsers without affecting other components.
- **Connector Process**: Facilitates communication between the annotator frontend and backend by sending data back and forth. Each backend requires a connector to maintain communication.

The main process of the annotator frontend also runs multiple threads to manage all processes and assign captioning tasks. The **Process Management** component is responsible for starting and stopping the processes, while the **Task Assigner** distributes tasks to available annotators based on backend availability and load. Currently, the frontend does not include a user interface and lacks a system monitor for real-time performance and load tracking. We welcome contributions to improve this system, including adding features or enhancing existing ones. **PRs are welcome!**

## Preparation for Running the System

Before running the system, ensure the following preparations are made:

### Collecting Raw Image Patches and OSM Data

Run the [scripts](../download) to collect image patches and raw OSM data. This will generate the `metadata.db` and `osm.db` files, which are required for the subsequent steps.

### Prepare the OSM Wiki Database

Download the OSM taginfo Wiki [database](https://taginfo.openstreetmap.org/sources/wiki), which is an SQLite database used for tag interpretation during the annotation process.

### Setting up the Annotation Database

The annotation process requires a database to store all annotations, annotator information, and prompt templates. You can create this database using the data builder [notebook](../tools/database_builder.ipynb).

Once the database is set up, you need to add some initial data:

1. **Set up the `annotator` table**: Add annotator names in the `ANNOTATOR` column. This determines which annotators are available for the annotation process. The names should match those listed in the `_VALID_MODELS` in the `annotation/annotators` directory, representing the backend annotators that will be used.

2. **Set up the `prompt` table**: Prepare the prompt templates for the captioning task. Some examples can be found [here](../docs/prompt_templates.md). Make sure to properly include placeholders for different types of prompts as shown in the example templates. Add the template to the `PROMPT` column and specify the prompt type in the `TYPE` column. We currently support three prompt types:
   - `2`: `area`
   - `3`: `non-area`
   - `11`: `caption revision`

3. **Set up the `annotator_prompt` table**: Create the mapping between annotators and prompts. Add the mapping in the `ANNOTATOR` and `PROMPT` columns, both referencing the `ID` columns in the `annotator` and `prompt` tables, respectively.

During the annotation process, the data producer selects a random valid prompt template and formats it into a "task." The task is passed to the task assigner, which identifies available annotators via the `annotator_prompt` table and assigns the task accordingly. The annotator backend processes the task, and the response is saved to the database.

### Setting up Revision Database (Optional)

If you plan to use the system for caption augmentation, you will need an additional database to store raw captions and their corresponding revisions. You can create this revision database using the data builder [notebook](../tools/database_builder.ipynb). The revision database has two tables to set up:

1. **Set up the `rewrite_raw` table**: This table stores raw captions and requires manual input for three columns: `patch_id`, `prompt_id`, and `text`.

   - The `patch_id` corresponds to the image patch ID in the `patch` table of the metadata database.
   - The `prompt_id` corresponds to the ID of the prompt in the `prompt` table of the annotation database.
   - The `text` column contains the raw caption.

2. **Set up the `rewrite_examples` table**: This table stores caption revisions and requires manual input for two columns: `rewrite_raw_id` and `rewrite_text`.

   - The `rewrite_raw_id` is the ID of the raw caption from the `rewrite_raw` table.
   - The `rewrite_text` column contains the revision of the raw caption.

## Running the System

### Step 1: Start the Annotator Backend

The annotator backend must be started before the frontend. Start the backend by running the `annotator_backend.py` script. Example command:

```bash
#!/bin/bash
python annotator_backend.py \
--annotator MixtralAnnotator \
--model-id mistralai/Mistral-Nemo-Instruct-2407 \
--port 5000
```

The `annotator` argument specifies the backend to use, such as `MixtralAnnotator`. This refers to the class name of the backend in the `annotators` directory. Developers can add additional backends by creating new classes in the `annotators` directory that inherit from `BaseAnnotator` and registering them in the `annotators/__init__.py` file. Currently, all annotators are hosted locally via vLLM, but you can also write code to run on CPU or use an API. Frameworks like LangChain or GraphRAG can also be implemented as annotator backends to support more advanced agents.

The `model-id` argument specifies the model ID used by the backend. In this case, it's `mistralai/Mistral-Nemo-Instruct-2407`, which is the model performing the captioning task.

The `port` argument specifies the port on which the backend will run. Each backend should use a unique port.

### Step 2: Start the Annotator Frontend

Once the annotator backends are running, start the annotator frontend by running the `annotator_frontend.py` script. Example command:

```bash
#!/bin/bash
python annotator_frontend.py \
--db_root ../database \
--fetch_size=50 \
--map_element_threshold 2 \
--num_data_producer 4 
```

The `db_root` argument specifies the root directory containing all necessary databases:

- `metadata.db`: Metadata database for image patch collection.
- `osm.db`: OSM database for raw OSM data collection.
- `annotation.db`: Annotation database for storing all annotations, annotator information, and prompt templates.
- `annotation_meta.db` (optional): Database for the revision process.
- `taginfo-wiki.db`: OSM taginfo Wiki database for tag interpretation.

The `fetch_size` argument defines the number of rows to fetch at a time for the data producer. To optimize performance, we use a batch reading mechanism to reduce the number of database queries and cache the data in memory.

The `map_element_threshold` argument specifies the minimum number of tags for an OSM element to be considered valid. Invalid elements are filtered out before the data producer reads the data.

The `num_data_producer` argument specifies the number of data producer processes to run. Each data producer process reads raw OSM data and interprets it into formatted prompts for the annotator backend, and assigns tasks to the available annotators.

>Note: You will need to modify lines `283` to `286` in `annotator_frontend.py` to match the actual annotator backends. This is hardcoded for now and may be improved in the future.
