from gcpaiutils.train import TrainJobHandler, TrainJobSpecHandler
from gcpaiutils.predict import ScoreJobHandler, ScoreJobSpecHandler
from gcpaiutils.postprocess import PostprocessJobHandler, PostprocessJobSpecHandler
from gcpaiutils.preprocess import PreprocessJobHandler, PreprocessJobSpecHandler
from gcpaiutils.utils import get_model_path_from_info_path, get_deployment_config, get_hardware_config,\
    get_user, get_problem, get_version, get_gcs_credentials, make_temp_dir, get_job_assessment,\
    get_selector, get_metadata, get_model_metadata
from googleapiclient import discovery
from google.cloud import storage
from shutil import rmtree
import logging
from google.oauth2.service_account import Credentials
from time import sleep
import os
import json

logger = logging.getLogger("OperatorsLogger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

TIME_INTERVAL = 60*1


def poll(deployment_config, time_interval, jobs):
    """
    Monitors job status on GCP AI Platform.

    :param deployment_config: YAML file containing all deployment variables
    :param time_interval: interval (in seconds) between two consecutive checks
    :param jobs: list of jobs to monitor
    :return: dict containing job names as keys and job status (SUCCEEDED, FAILED) as value
    """

    GLOBALS = get_deployment_config(deployment_config)
    try:
        ai_credentials = Credentials.from_service_account_file(GLOBALS["AI_PLATFORM_SA"])
    except:
        ai_credentials = None

    mlapi = discovery.build('ml', 'v1', credentials=ai_credentials, cache_discovery=False)

    still_running = True
    while still_running:
        status = {}
        states = 1
        for job in jobs:
            request = mlapi.projects().jobs().get(name='projects/' + GLOBALS["PROJECT_ID"] + '/jobs/' + job)
            counter = 0
            done = False
            while counter < 10 and done is False:
                try:
                    jobs_info = request.execute()  # manage retries for HTTP errors. Should design a decorator.
                    done = True
                except:
                    sleep(60)
                    counter += 1
            status[job] = jobs_info['state']
            states *= (jobs_info['state'] in ["SUCCEEDED", "FAILED"])
        if states:
            still_running = False
        else:
            logging.info("Waiting for jobs:")
            for key, value in status.items():
                if value not in ["SUCCEEDED", "FAILED"]:
                    logging.info(key)
            sleep(time_interval)
    return status


def train(deployment_config, atom=None, atom_params=None, hyperspace=None, **kwargs):
    """
    Submits train job to GCP AI Platform and waits for completion. Compute is done remotely.

    :param deployment_config: YAML file containing all deployment variables
    :param atom: algorithm name as tagged in Container Registry (atom)
    :param atom_params: user-specified parameters to configurate atom (useful to overwrite atom defaults with no tuning)
    :param hyperspace: hyper-parameter tuning configuration
    :param kwargs:
    :return:
    """
    _globals = get_deployment_config(deployment_config)
    hypertune = hyperspace is not None
    model_dir = f"gs://{_globals['MODEL_BUCKET_NAME']}/{get_user(kwargs)}/ACTIVE_MODELS/{get_problem(kwargs)}/"
    train_files = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='data_uri')

    trainingInput = {
        "trainFiles": train_files,
        "modelDir": model_dir,
        "scaleTier": "CUSTOM"
    }

    # Add user-specified parameters
    if isinstance(atom_params, dict):
        trainingInput['args'] = []
        for k, v in atom_params.items():
            trainingInput['args'] += [k, str(v)]

    # Get metadata file
    metadata = get_metadata(_globals, 'TRAIN', kwargs)

    submitted_jobs = []
    trainingInput["masterType"] = get_hardware_config(atom=atom, data_size=metadata['size'], scoring=False)

    if hypertune:
        trainingInput["hyperparameters"] = hyperspace[atom]
        trainingInput["hypertuneLoss"] = hyperspace[atom]["hyperparameterMetricTag"]

    S = TrainJobSpecHandler(deployment_config=deployment_config, algorithm=atom, inputs=trainingInput,
                            hypertune=hypertune, request_ids={'user': get_user(kwargs),
                                                              'problem': get_problem(kwargs),
                                                              'version': get_version(kwargs)})
    S.create_job_specs()
    T = TrainJobHandler(deployment_config=deployment_config, job_executor='mlapi')
    T.submit_job(S.job_specs)
    if T.success:
        submitted_jobs.append(S.job_specs['jobId'])
        logging.info("Train request successful: {}".format(S.job_specs['jobId']))
    else:
        raise ValueError("Unable to submit train job.")

    status = poll(deployment_config, TIME_INTERVAL, submitted_jobs)
    kwargs['task_instance'].xcom_push(key='successful_jobs', value=get_job_assessment(status))


def new_selection_from_folder(deployment_config=None, selector_class_dict=None, **kwargs):
    """
    Selects among trained models found in MODELS folder those that will make it to production. Compute is done locally.

    :param deployment_config: YAML file containing all deployment variables
    :param selector_class_dict: dict of customized BaseSelector object. Dict keys are strategy names.
    :param kwargs:
    :return:
    """
    _globals = get_deployment_config(deployment_config)

    evaluation_metric = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='evaluation_metric')

    # Retrieve blob list from MODELS folder
    gcs_client = storage.Client(project=_globals['PROJECT_ID'])
    gcs_bucket = gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"])

    # Import from GCS
    path_prefix = os.path.join(get_user(kwargs), "ACTIVE_MODELS", get_problem(kwargs))
    gcs_all_blobs = list(gcs_bucket.list_blobs(prefix=path_prefix))
    gcs_info_blob_list = [item for item in gcs_all_blobs if item.name.split("/")[-1].startswith("info")]
    gcs_stratified_info_blob_list = [item for item in gcs_all_blobs if item.name.split("/")[-1].startswith("stratified_info")]
    gcs_blob_list = gcs_info_blob_list + gcs_stratified_info_blob_list

    # Call a selection method
    job_list = set([item.name.split("/")[-2] for item in gcs_blob_list])
    info_dir = make_temp_dir(os.getcwd())
    for job in job_list:
        # download in dir with job name
        tmp_dir_name = "{}/{}".format(info_dir, job)
        os.mkdir(tmp_dir_name)

        for gcs_source_blob in [blob for blob in gcs_blob_list if job in blob.name]:
            local_destination = os.path.join(tmp_dir_name, gcs_source_blob.name.split("/")[-1])
            gcs_source_blob.download_to_filename(local_destination, client=gcs_client)  # TODO: make this multithread

        # concatenate file name to match GCS flat namespace
        for file in [f for f in os.listdir(tmp_dir_name) if os.path.isfile(os.path.join(tmp_dir_name, f))]:
            os.rename(os.path.join(tmp_dir_name, file),
                      os.path.join(info_dir, '_'.join([job, file])))

        # remove dir
        rmtree(tmp_dir_name)

    # make sure selector_class_dict is imported in this module
    root_dest_uri = f"gs://{_globals['MODEL_BUCKET_NAME']}/{get_user(kwargs)}/SELECTOR/{get_problem(kwargs)}/"

    selected_info = {}
    for key, d in selector_class_dict.items():
        S = d['selector'](deployment_config=deployment_config, model_dir=info_dir, evaluation_metric=evaluation_metric,
                          problem_type='classification', n_class=5, verbose=True)
        dest_uri = root_dest_uri + key + "/"  # dict key is strategy name
        if 'kwargs' not in d.keys():
            d['kwargs'] = {}
        selected_info[key] = S.select(destination_uri=dest_uri, validation_schema=d['validation_schema'], **d['kwargs'])
    rmtree(info_dir)

    kwargs['task_instance'].xcom_push(key='selected_info', value=selected_info)


def selection_from_folder(deployment_config, selector_class_dict=None, **kwargs):
    """
    Selects among trained models found in MODELS folder those that will make it to production. Compute is done locally.

    :param deployment_config: YAML file containing all deployment variables
    :param selector_class_dict: dict of customized BaseSelector object. Dict keys are strategy names.
    :param kwargs:
    :return:
    """
    _globals = get_deployment_config(deployment_config)

    evaluation_metric = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='evaluation_metric')

    # Retrieve blob list from MODELS folder
    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)
    gcs_bucket = gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"])

    # Import from GCS
    path_prefix = os.path.join(get_user(kwargs), get_problem(kwargs), get_version(kwargs), "MODELS")
    gcs_all_blobs = list(gcs_bucket.list_blobs(prefix=path_prefix))
    gcs_info_blob_list = [item for item in gcs_all_blobs if item.name.split("/")[-1].startswith("info")]
    gcs_stratified_info_blob_list = [item for item in gcs_all_blobs if item.name.split("/")[-1].startswith("stratified_info")]
    gcs_blob_list = gcs_info_blob_list + gcs_stratified_info_blob_list

    # Call a selection method
    job_list = set([item.name.split("/")[-2] for item in gcs_blob_list])
    info_dir = make_temp_dir(os.getcwd())
    for job in job_list:
        # download in dir with job name
        tmp_dir_name = "{}/{}".format(info_dir, job)
        os.mkdir(tmp_dir_name)

        for gcs_source_blob in [blob for blob in gcs_blob_list if job in blob.name]:
            local_destination = os.path.join(tmp_dir_name, gcs_source_blob.name.split("/")[-1])
            gcs_source_blob.download_to_filename(local_destination, client=gcs_client)  # TODO: make this multithread

        # concatenate file name to match GCS flat namespace
        for file in [f for f in os.listdir(tmp_dir_name) if os.path.isfile(os.path.join(tmp_dir_name, f))]:
            os.rename(os.path.join(tmp_dir_name, file),
                      os.path.join(info_dir, '_'.join([job, file])))

        # remove dir
        rmtree(tmp_dir_name)

    # make sure selector_class_dict is imported in this module
    root_dest_uri = f"gs://{_globals['MODEL_BUCKET_NAME']}/{get_user(kwargs)}/SELECTOR/{get_problem(kwargs)}/"

    selected_info = {}
    for key, d in selector_class_dict.items():
        S = d['selector'](deployment_config=deployment_config, model_dir=info_dir, evaluation_metric=evaluation_metric,
                problem_type='classification', n_class=5, verbose=True)
        dest_uri = root_dest_uri + key + "/"  # dict key is strategy name
        if 'kwargs' not in d.keys():
            d['kwargs'] = {}
        selected_info[key] = S.select(destination_uri=dest_uri, validation_schema=d['validation_schema'], **d['kwargs'])
    rmtree(info_dir)
    kwargs['task_instance'].xcom_push(key='selected_info', value=selected_info)


def selection(deployment_config, train_task_ids=None, selector_class_dict=None, **kwargs):
    """
    Selects among trained models those that will make it to production. Compute is done locally.

    :param deployment_config: YAML file containing all deployment variables
    :param train_task_ids: list of training tasks that represent the universe on which selection takes place
    :param selector_class_dict: dict of customized BaseSelector object. Dict keys are strategy names.
    :param kwargs:
    :return:
    """
    _globals = get_deployment_config(deployment_config)

    evaluation_metric = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='evaluation_metric')

    successful_train_jobs = []
    for train_task in train_task_ids:
        current_train_job = kwargs['task_instance'].xcom_pull(task_ids=train_task, key='successful_jobs')
        if current_train_job is None:
            continue  # this happens when a train_task is skipped
        successful_train_jobs.append(current_train_job)
    successful_train_jobs = [item for sublist in successful_train_jobs for item in sublist]  # flatten

    # Call a selection method
    info_dir = make_temp_dir(os.getcwd())
    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)
    gcs_bucket = gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"])

    for job in successful_train_jobs:
        # download in dir with job name
        tmp_dir_name = "{}/{}".format(info_dir, job.replace("train_", ""))
        os.mkdir(tmp_dir_name)

        # Import from GCS
        path_prefix = os.path.join(get_user(kwargs), get_problem(kwargs), get_version(kwargs), "MODELS",
                                   job.replace("train_", ""))
        gcs_info_blob_list = list(gcs_bucket.list_blobs(prefix=os.path.join(path_prefix, "info")))
        gcs_stratified_info_blob_list = list(gcs_bucket.list_blobs(prefix=os.path.join(path_prefix, "stratified_info")))
        gcs_blob_list = gcs_info_blob_list + gcs_stratified_info_blob_list

        for gcs_source_blob in gcs_blob_list:
            local_destination = os.path.join(info_dir, job.replace("train_", ""), gcs_source_blob.name.split("/")[-1])
            gcs_source_blob.download_to_filename(local_destination, client=gcs_client)  # TODO: make this multithread

        # concatenate file name to match GCS flat namespace
        for file in [f for f in os.listdir(tmp_dir_name) if os.path.isfile(os.path.join(tmp_dir_name, f))]:
            os.rename(os.path.join(tmp_dir_name, file),
                      os.path.join(info_dir, '_'.join([job.replace("train_", ""), file])))

        # remove dir
        rmtree(tmp_dir_name)

    # make sure selector_class_dict is imported in this module
    root_dest_uri = f"gs://{_globals['MODEL_BUCKET_NAME']}/{get_user(kwargs)}/SELECTOR/{get_problem(kwargs)}/"

    selected_info = {}
    for key, d in selector_class_dict.items():
        S = d['selector'](deployment_config=deployment_config, model_dir=info_dir, evaluation_metric=evaluation_metric,
                problem_type='classification', n_class=5, verbose=True)
        dest_uri = root_dest_uri + key + "/"  # dict key is strategy name
        if 'kwargs' not in d.keys():
            d['kwargs'] = {}
        selected_info[key] = S.select(destination_uri=dest_uri, validation_schema=d['validation_schema'], **d['kwargs'])
    rmtree(info_dir)
    kwargs['task_instance'].xcom_push(key='selected_info', value=selected_info)


def score(deployment_config, use_proba=None, **kwargs):
    """
    Submits score job(s) to GCP AI Platform and waits for completion. Compute is done remotely.

    :param deployment_config: YAML file containing all deployment variables
    :param score_dir: URI where to write results
    :param use_proba: (string). 0 = No / 1 = Yes
    :param master_type: GCP VM type to use during scoring
    :param kwargs:
    :return:
    """

    _globals = get_deployment_config(deployment_config)

    # Get metadata file
    metadata = get_metadata(_globals, 'SCORE', kwargs)
    score_dir = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='data_uri')
    data_consistent_models = kwargs['task_instance'].xcom_pull(task_ids='metadata_check', key='data_consistent_models')

    scoreInput = {
        "scoreDir": score_dir,
        "useProba": use_proba,
        "scaleTier": "CUSTOM"
    }

    local_dir, local_destinations = get_selector(_globals, kwargs)

    selected_info = {}
    for destination in local_destinations:

        with open(destination, 'r') as f:
            selector_dict = json.load(f)
        selected_info[destination.split("/")[-2]] = selector_dict['selection']  # key == strategy name

    rmtree(local_dir)

    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)
    gcs_bucket = gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"])
    submitted_scoring_jobs = {}
    for strategy_name, value in selected_info.items():
        for info in value:

            model_path = get_model_path_from_info_path(info)
            logger.info(f"Model path: {algo}")
            if model_path.split("/")[-1] not in data_consistent_models:
                logger.warning(f"Skipping model {model_path.split('/')[-1]} because currert data is insufficient to yield useful predictions.")
                continue # skip model if current available data is insufficient to yield useful predictions

            currentInput = scoreInput.copy()
            algo = '_'.join(model_path.split("/")[0].split("_")[4:])
            blobs = list(gcs_bucket.list_blobs(prefix=os.path.join(get_user(kwargs), "ACTIVE_MODELS",
                                                                   get_problem(kwargs), model_path)))  # unique id
            if len(blobs) == 1:
                # model is a file
                currentInput["modelFile"] = os.path.join(_globals["MODEL_BUCKET_ADDRESS"], blobs[0].name)
            elif len(blobs) > 1:
                # model is not a file but a folder (multiple files)
                currentInput["modelFile"] = os.path.join(_globals["MODEL_BUCKET_ADDRESS"],
                                                         '/'.join(blobs[0].name.split("/")[0:-1]))
            else:
                raise FileNotFoundError("Could not find any blob matching %s" % os.path.join(model_path))

            currentInput["outputDir"] = f"gs://{_globals['MODEL_BUCKET_NAME']}/{get_user(kwargs)}/{get_problem(kwargs)}/RESULTS_STAGING/{strategy_name}/{model_path.split('/')[0]}/"

            currentInput["masterType"] = get_hardware_config(atom=algo, data_size=metadata['size'], scoring=True)

            S = ScoreJobSpecHandler(algorithm=algo,
                                    deployment_config=deployment_config,
                                    inputs=currentInput, request_ids={'user': get_user(kwargs),
                                                                      'problem': get_problem(kwargs),
                                                                      'version': ''})
            S.create_job_specs()
            T = ScoreJobHandler(deployment_config=deployment_config, job_executor='mlapi')
            T.submit_job(S.job_specs)
            if T.success:
                submitted_scoring_jobs[S.job_specs['jobId']] = model_path.split("/")[0]  # corresponding train job id
                logging.info("Score request successful: {}".format(S.job_specs['jobId']))
            else:
                raise ValueError("Unable to submit score job.")

    if not submitted_scoring_jobs:
        raise ValueError("No jobs selected for scoring.")

    # Retrieve scoring
    status = poll(deployment_config, TIME_INTERVAL, submitted_scoring_jobs)
    kwargs['task_instance'].xcom_push(key='successful_jobs', value=get_job_assessment(status))


def aggregate(deployment_config, neutralized=False, **kwargs):
    """
    Aggregate model scoring. Compute is done remotely.

    :param deployment_config:
    :param kwargs:
    :return:
    """

    _globals = get_deployment_config(deployment_config)
    local_dir, local_destinations = get_selector(_globals, kwargs)

    selected_info = {}
    for destination in local_destinations:
        with open(destination, 'r') as f:
            selector_dict = json.load(f)
        selected_info[destination.split("/")[-2]] = selector_dict
    rmtree(local_dir)

    # Get metadata file
    metadata = get_metadata(_globals, 'SCORE', kwargs)

    scoreInput = {
        "masterType": get_hardware_config(atom='aggregator', data_size=metadata['size']),
        "scaleTier": "CUSTOM"
    }

    root_output_dir = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='output_uri')
    staging_dir = "RESULTS_STAGING"
    if neutralized:
        root_output_dir = "/".join(["/".join(root_output_dir.split("/")[0:-2]),
                                   "NEUTRALIZED_" + root_output_dir.split("/")[-2]]) + "/"
        staging_dir = "NEUTRALIZED_RESULTS_STAGING"

    submitted_postprocess_jobs_list = []
    for strategy_name, value in selected_info.items():
        if not value['selection']:
            continue
        if value['aggregation'] == 'average':

            current_score_input = scoreInput.copy()

            current_score_input["scoreDir"] = "gs://{}/{}/{}/{}/{}/{}/".format(_globals["MODEL_BUCKET_NAME"],
                                                                                            get_user(kwargs),
                                                                                            get_problem(kwargs),
                                                                                            get_version(kwargs),
                                                                                            staging_dir,
                                                                                         strategy_name)

            current_score_input["outputDir"] = root_output_dir + strategy_name + "/"

            S = PostprocessJobSpecHandler(algorithm='aggregator',
                                          deployment_config=deployment_config,
                                          inputs=current_score_input, request_ids={'user': get_user(kwargs),
                                                                                   'problem': get_problem(kwargs),
                                                                                   'version': get_version(kwargs)})
            S.create_job_specs()
            T = PostprocessJobHandler(deployment_config=deployment_config, job_executor='mlapi')
            T.submit_job(S.job_specs)
            submitted_postprocess_jobs = {}
            if T.success:
                submitted_postprocess_jobs[S.job_specs['jobId']] = S.job_specs['jobId']  # corresponding train job id
                logging.info("Postprocess request successful: {}".format(S.job_specs['jobId']))
            else:
                raise ValueError("Unable to submit postprocess job.")

            submitted_postprocess_jobs_list.append(submitted_postprocess_jobs)
        else:
            raise NotImplementedError("Only supported aggregation is: 'average'")

    # Retrieve scoring
    status_list = []
    for item in submitted_postprocess_jobs_list:
        status = poll(deployment_config, TIME_INTERVAL, item)
        status_list.append(get_job_assessment(status))
    kwargs['task_instance'].xcom_push(key='successful_jobs', value=status_list)


def algorithm_routing(deployment_config, algorithm_space, **kwargs):

    _globals = get_deployment_config(deployment_config)

    # Get metadata file
    metadata = get_metadata(_globals, 'TRAIN', kwargs)

    # Assess NULL values presence
    null_presence = False
    for key, value in metadata['missing_data_rate'].items():
        if value > 0:
            null_presence = True
            break

    # Route training tasks
    tasks_to_trigger = list(algorithm_space.keys())
    for algorithm, value in algorithm_space.items():
        if null_presence:
            if not value['null_compatible']:
                tasks_to_trigger.remove(algorithm)

    return tasks_to_trigger


def metadata_check(deployment_config, information_loss_tolerance=0.1, **kwargs):

    _globals = get_deployment_config(deployment_config)

    trained_model_metadata = get_model_metadata(_globals, kwargs)
    current_data_metadata = get_metadata(_globals, None, kwargs)

    # Check if metadata linked to trained models matches metadata from current data
    missing_features_pct = {}
    for key, model_featimp in trained_model_metadata.items():
        relevant_features = list(model_featimp.loc[model_featimp['feature_importance'] > 0]['feature_name'])
        missing_importance = 0
        for feature in relevant_features:
            if current_data_metadata['missing_data_rate'][feature] > 0:
                missing_importance += model_featimp.loc[model_featimp['feature_name'] == feature]['feature_importance']
        missing_features_pct[key.replace("featimp", "model").replace("csv", "pkl")] = missing_importance

    data_consistent_models = []
    for model, pct in missing_features_pct.items():
        if pct <= information_loss_tolerance:
            data_consistent_models.append(model)

    kwargs['task_instance'].xcom_push(key='data_consistent_models', value=data_consistent_models)


def data_evaluation(deployment_config, data_uri, user, problem, **kwargs):

    _globals = get_deployment_config(deployment_config)
    model_dir = "gs://{}/{}/{}/METADATA/".format(_globals["MODEL_BUCKET_NAME"], user, problem)

    preprocess_input = {'trainFiles': data_uri,
                        'modelDir': model_dir,
                        'scaleTier': 'CUSTOM',
                        'masterType': 'n1-highmem-8' # 8 vCPUs / 52 GB RAM  --- 'n1-highmem-4'  # 4 vCPUs / 26 GB RAM
                        }

    S = PreprocessJobSpecHandler(deployment_config=deployment_config,
                                 algorithm='data_evaluator',
                                 append_job_id=False,  # ensure you overwrite same destination
                                 inputs=preprocess_input,
                                 request_ids={'user': user, 'problem': problem, 'version': ''})
    S.create_job_specs()
    T = PreprocessJobHandler(deployment_config=deployment_config, job_executor='mlapi')
    T.submit_job(S.job_specs)
    submitted_preprocess_jobs = {}
    if T.success:
        submitted_preprocess_jobs[S.job_specs['jobId']] = S.job_specs['jobId']  # corresponding train job id
        logging.info("Preprocess request successful: {}".format(S.job_specs['jobId']))
    else:
        raise ValueError("Unable to submit preprocess job.")

    # Retrieve scoring
    status = poll(deployment_config, TIME_INTERVAL, submitted_preprocess_jobs)
    kwargs['task_instance'].xcom_push(key='successful_jobs', value=get_job_assessment(status))


def wait_dag_status(deployment_config, dag_type, conf, **kwargs):

    _globals = get_deployment_config(deployment_config)

    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)

    while True:
        sleep(60)
        gcs_blob_list = list(gcs_client.list_blobs(bucket_or_name=_globals["MODEL_BUCKET_NAME"],
                                                   prefix=os.path.join(conf['user'], conf['problem'], "STATUS")))

        if len(gcs_blob_list) > 1:
            raise ValueError("Found more than one status file")
        elif len(gcs_blob_list) == 1:
            if gcs_blob_list[0].name.split("/")[-1] == 'success.json':
                break
            elif gcs_blob_list[0].name.split("/")[-1] == 'failure.json':
                raise ValueError("DAG failed")
            else:
                raise ValueError("Unknown status file: %s" % gcs_blob_list[0].name.split("/")[-1])


def clear_dag_status(deployment_config, dag_type, conf, **kwargs):

    _globals = get_deployment_config(deployment_config)
    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)

    gcs_blob_list = list(gcs_client.list_blobs(bucket_or_name=_globals["MODEL_BUCKET_NAME"],
                                               prefix=os.path.join(conf['user'], conf['problem'], "STATUS")))

    for blob in gcs_blob_list:
        blob.delete()


def notify_dag_status(deployment_config, dag_type, status, **kwargs):

    _globals = get_deployment_config(deployment_config)

    status_dir = make_temp_dir(os.getcwd())
    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)

    local_status_file = os.path.join(status_dir, '{}.json'.format(status))
    with open(local_status_file, 'w') as f:
        json.dump('0', f)

    gcs_destination_bucket = gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"])
    if dag_type == 'TRAIN':
        gcs_destination_blob = '/'.join([get_user(kwargs), "ACTIVE_MODELS", get_problem(kwargs)
                                            , "STATUS", local_status_file.split("/")[-1]])
    elif dag_type == 'SCORE':
        gcs_destination_blob = '/'.join([get_user(kwargs), get_problem(kwargs)
                                        , "STATUS", local_status_file.split("/")[-1]])
    else:
        raise ValueError(f"dag_type {dag_type} not recognized. Must be either TRAIN or SCORE.")
    b = storage.blob.Blob(gcs_destination_blob, gcs_destination_bucket)
    b.upload_from_filename(local_status_file, client=gcs_client)

    client_output_uri = kwargs['task_instance'].xcom_pull(task_ids=['retrieve_params'], key='output_uri')[0]
    if client_output_uri is not None:
        # Notify client
        client_output_uri_shards = client_output_uri.split("/")
        client_bucket_name = client_output_uri_shards[2]
        gcs_client_destination_bucket = gcs_client.get_bucket(client_bucket_name)
        gcs_destination_blob = '/'.join(client_output_uri_shards[3:-1] + [local_status_file.split("/")[-1]])
        b = storage.blob.Blob(gcs_destination_blob, gcs_client_destination_bucket)
        b.upload_from_filename(local_status_file, client=gcs_client)

    rmtree(status_dir)
