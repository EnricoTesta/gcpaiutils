from gcpaiutils.train import TrainJobHandler, TrainJobSpecHandler
from gcpaiutils.predict import ScoreJobHandler, ScoreJobSpecHandler
from gcpaiutils.postprocess import PostprocessJobHandler, PostprocessJobSpecHandler
from gcpaiutils.preprocess import PreprocessJobHandler, PreprocessJobSpecHandler
from gcpaiutils.utils import get_model_path_from_info_path, get_deployment_config, get_hardware_config,\
    get_user, get_problem, get_version, get_gcs_credentials, make_temp_dir, get_job_assessment, get_selector
from googleapiclient import discovery
from google.cloud import storage
from subprocess import check_call
from shutil import rmtree
from pandas import read_csv, concat
import logging
from google.oauth2.service_account import Credentials
from time import sleep
import os
import json

logging.getLogger("OperatorsLogger")
logging.getLogger("OperatorsLogger").setLevel(logging.INFO)

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


def train(deployment_config, atom=None, train_files=None, master_type=None, hyperspace=None, **kwargs):
    """
    Submits train job to GCP AI Platform and waits for completion.

    :param deployment_config: YAML file containing all deployment variables
    :param atom: algorithm name as tagged in Container Registry (atom)
    :param train_files: .csv file(s) complete URI
    :param master_type: GCP VM type to use during training
    :param hyperspace: hyper-parameter tuning configuration
    :param kwargs:
    :return:
    """
    hypertune = hyperspace is not None
    trainingInput = {
        "trainFiles": train_files,
        "scaleTier": "CUSTOM"
    }

    submitted_jobs = []

    current_train_input = trainingInput.copy()
    current_train_input["masterType"] = master_type

    if hypertune:
        current_train_input["hyperparameters"] = hyperspace[atom]
        current_train_input["hypertuneLoss"] = hyperspace[atom]["hyperparameterMetricTag"]

    S = TrainJobSpecHandler(deployment_config=deployment_config, algorithm=atom, inputs=current_train_input,
                            hypertune=hypertune)
    S.create_job_specs()
    T = TrainJobHandler(deployment_config=deployment_config, job_executor='mlapi')
    T.submit_job(S.job_specs)
    if T.success:
        submitted_jobs.append(S.job_specs['jobId'])
        logging.info("Train request successful: {}".format(S.job_specs['jobId']))
    else:
        raise ValueError("Unable to submit train job.")

    status = poll(deployment_config, TIME_INTERVAL, submitted_jobs)

    successful_jobs = [job for job, s in status.items() if s == 'SUCCEEDED']
    failed_jobs = [job for job, s in status.items() if s == 'FAILED']
    if len(failed_jobs) >= 1:
        for job in failed_jobs:
            logging.error("Train job failed: {}".format(job))
        raise ValueError("One or more train jobs failed")
    task_instance = kwargs['task_instance']
    task_instance.xcom_push(key='successful_jobs', value=successful_jobs)


def selection(deployment_config, train_task_ids=None, selector_class=None, **kwargs):
    """
    Selects among trained models those that will make it to production.

    :param deployment_config: YAML file containing all deployment variables
    :param train_task_ids: list of training tasks that represent the universe on which selection takes place
    :param selector_class: customized BaseSelector object
    :param kwargs:
    :return:
    """
    GLOBALS = get_deployment_config(deployment_config)

    successful_train_jobs = []
    for train_task in train_task_ids:
        successful_train_jobs.append(kwargs['task_instance'].xcom_pull(task_ids=train_task, key='successful_jobs'))
    successful_train_jobs = [item for sublist in successful_train_jobs for item in sublist]  # flatten

    # Call a selection method
    info_dir = "{}/info_tmp".format(os.getcwd())
    if os.path.exists(info_dir):
        rmtree(info_dir)
    os.mkdir(info_dir)

    for job in successful_train_jobs:
        # download in dir with job name
        tmp_dir_name = "{}/{}".format(info_dir, job.replace("train_", ""))
        os.mkdir(tmp_dir_name)
        # TODO: need to configure gsutil with new deployment
        cmd = 'gsutil -m cp {}{}/info_*.csv {}/{}'.format(GLOBALS["MODEL_BUCKET_ADDRESS"],
                                                           job.replace("train_", ""), info_dir,
                                                           job.replace("train_", ""))
        os.system(cmd)

        # concatenate file name to match GCS flat namespace
        for file in [f for f in os.listdir(tmp_dir_name) if os.path.isfile(os.path.join(tmp_dir_name, f))]:
            os.rename(os.path.join(tmp_dir_name, file),
                      os.path.join(info_dir, '_'.join([job.replace("train_", ""), file])))

        # remove dir
        rmtree(tmp_dir_name)

    # make sure selector_class is imported in this module
    S = selector_class(model_dir=info_dir, problem_type='classification', n_class=5, verbose=True)
    selected_info = S.select()
    rmtree(info_dir)
    kwargs['task_instance'].xcom_push(key='selected_info', value=selected_info)


def score(deployment_config, selection_task_id=None, score_dir=None, use_proba=None, master_type=None,
          subject=None, problem=None, version=None, **kwargs):
    """
    Submits score job(s) to GCP AI Platform and waits for completion.

    :param deployment_config: YAML file containing all deployment variables
    :param selection_task_id: tasks that contains models to score
    :param score_dir: URI where to write results
    :param use_proba: (string). 0 = No / 1 = Yes
    :param master_type: GCP VM type to use during scoring
    :param subject: (string)
    :param problem: (string)
    :param version: (string)
    :param kwargs:
    :return:
    """
    task_instance = kwargs['task_instance']
    selected_info = task_instance.xcom_pull(task_ids=selection_task_id, key='selected_info')

    scoreInput = {
        "scoreDir": score_dir,
        "useProba": use_proba,
        "masterType": master_type,
        "scaleTier": "CUSTOM"
    }

    GLOBALS = get_deployment_config(deployment_config)
    submitted_scoring_jobs = {}

    for info in selected_info:

        currentInput = scoreInput.copy()
        model_path = get_model_path_from_info_path(info)

        # Look for model in appropriate blob (temporary - assumes single model in blob)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/gauth/z-account-main-project-0-ai-platform-default.json'  # TODO: use OAuth2 credentials
        # storage_client = storage.Client(credentials=storage_credentials)
        storage_client = storage.Client()
        blobs = list(storage_client.list_blobs(GLOBALS["MODEL_BUCKET_NAME"], prefix=model_path))  # unique id
        currentInput["modelFile"] = os.path.join(GLOBALS["MODEL_BUCKET_ADDRESS"], blobs[0].name)
        currentInput["outputDir"] = "gs://{}/{}/{}/RESULTS/{}/{}/".format(GLOBALS["CORE_BUCKET_NAME"],
                                                                          subject,
                                                                          problem, version, model_path.split("/")[0])

        S = ScoreJobSpecHandler(algorithm='_'.join(model_path.split("/")[0].split("_")[4:-1]),
                                deployment_config=deployment_config,
                                inputs=currentInput)
        S.create_job_specs()
        T = ScoreJobHandler(deployment_config=deployment_config, job_executor='mlapi')
        T.submit_job(S.job_specs)
        if T.success:
            submitted_scoring_jobs[S.job_specs['jobId']] = model_path.split("/")[0]  # corresponding train job id
            logging.info("Score request successful: {}".format(S.job_specs['jobId']))
        else:
            raise ValueError("Unable to submit score job.")

    # Retrieve scoring
    status = poll(deployment_config, TIME_INTERVAL, submitted_scoring_jobs)
    successful_scoring_jobs = {}
    for job, s in status.items():
        if s == 'SUCCEEDED':
            successful_scoring_jobs[job] = submitted_scoring_jobs[job]  # keep reference to corresponding training job
        else:
            raise ValueError("One or more score jobs failed")

    pred_dir = "{}/pred_dir".format(os.getcwd())
    if os.path.exists(pred_dir):
        rmtree(pred_dir)
    os.mkdir(pred_dir)

    for _, train_job in successful_scoring_jobs.items():  # hp that all score jobs are successful and sorted
        cmd = "gsutil cp gs://{}/{}/{}/RESULTS/{}/{}/results.csv {}/{}.csv".format(GLOBALS["CORE_BUCKET_NAME"],
                                                                                   subject, problem,
                                                                                   version,
                                                                                   train_job,
                                                                                   pred_dir,
                                                                                   train_job)
        check_call(cmd, shell=True)

    kwargs['task_instance'].xcom_push(key='pred_dir', value=pred_dir)


def train_new(deployment_config, atom=None, hyperspace=None, **kwargs):
    """
    Submits train job to GCP AI Platform and waits for completion. Compute is done remotely.

    :param deployment_config: YAML file containing all deployment variables
    :param atom: algorithm name as tagged in Container Registry (atom)
    :param master_type: GCP VM type to use during training
    :param hyperspace: hyper-parameter tuning configuration
    :param kwargs:
    :return:
    """

    _globals = get_deployment_config(deployment_config)
    hypertune = hyperspace is not None
    model_dir = "gs://{}/{}/{}/{}/MODELS/".format(_globals["MODEL_BUCKET_NAME"],
                                                  get_user(kwargs), get_problem(kwargs), get_version(kwargs))
    train_files = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='data_uri')

    trainingInput = {
        "trainFiles": train_files,
        "modelDir": model_dir,
        "scaleTier": "CUSTOM"
    }

    # TODO: retrieve from file
    data_size = kwargs['task_instance'].xcom_pull(task_ids='data_evaluation', key='data_size')

    submitted_jobs = []

    trainingInput["masterType"] = get_hardware_config(atom, data_size)

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


def selection_new(deployment_config, train_task_ids=None, selector_class=None, **kwargs):
    """
    Selects among trained models those that will make it to production. Compute is done locally.

    :param deployment_config: YAML file containing all deployment variables
    :param train_task_ids: list of training tasks that represent the universe on which selection takes place
    :param selector_class: customized BaseSelector object
    :param kwargs:
    :return:
    """
    _globals = get_deployment_config(deployment_config)

    successful_train_jobs = []
    for train_task in train_task_ids:
        successful_train_jobs.append(kwargs['task_instance'].xcom_pull(task_ids=train_task, key='successful_jobs'))
    successful_train_jobs = [item for sublist in successful_train_jobs for item in sublist]  # flatten

    # Call a selection method
    info_dir = make_temp_dir(os.getcwd())
    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)

    for job in successful_train_jobs:
        # download in dir with job name
        tmp_dir_name = "{}/{}".format(info_dir, job.replace("train_", ""))
        os.mkdir(tmp_dir_name)

        # Import from GCS
        gcs_blob_list = list(gcs_client.list_blobs(bucket_or_name=_globals["MODEL_BUCKET_NAME"],
                                                   prefix=os.path.join(get_user(kwargs), get_problem(kwargs),
                                                                       get_version(kwargs), "MODELS",
                                                                       job.replace("train_", ""), "info")))

        for gcs_source_blob in gcs_blob_list:
            local_destination = os.path.join(info_dir, job.replace("train_", ""), gcs_source_blob.name.split("/")[-1])
            gcs_source_blob.download_to_filename(local_destination, client=gcs_client)  # TODO: make this multithread

        # concatenate file name to match GCS flat namespace
        for file in [f for f in os.listdir(tmp_dir_name) if os.path.isfile(os.path.join(tmp_dir_name, f))]:
            os.rename(os.path.join(tmp_dir_name, file),
                      os.path.join(info_dir, '_'.join([job.replace("train_", ""), file])))

        # remove dir
        rmtree(tmp_dir_name)

    # make sure selector_class is imported in this module
    S = selector_class(deployment_config=deployment_config, model_dir=info_dir,
                       problem_type='classification', n_class=5, verbose=True)
    dest_uri = "gs://{}/{}/{}/{}/SELECTOR/".format(_globals["MODEL_BUCKET_NAME"],
                                                   get_user(kwargs), get_problem(kwargs),
                                                   get_version(kwargs))
    selected_info = S.select(destination_uri=dest_uri)
    rmtree(info_dir)
    kwargs['task_instance'].xcom_push(key='selected_info', value=selected_info)


def score_new(deployment_config, use_proba=None, master_type=None, **kwargs):
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
    # TODO: retrieve data evaluation from file to assess hardware configuration and remove explicit master_type

    score_dir = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='data_uri')

    scoreInput = {
        "scoreDir": score_dir,
        "useProba": use_proba,
        "masterType": master_type,
        "scaleTier": "CUSTOM"
    }

    local_dir, local_destination = get_selector(_globals, kwargs)

    with open(local_destination, 'r') as f:
        selector_dict = json.load(f)
    selected_info = selector_dict['selection']

    rmtree(local_dir)

    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)
    submitted_scoring_jobs = {}
    for info in selected_info:

        currentInput = scoreInput.copy()
        model_path = get_model_path_from_info_path(info)

        # storage_client = storage.Client(credentials=storage_credentials)
        blobs = list(gcs_client.list_blobs(_globals["MODEL_BUCKET_NAME"],
                                           prefix=os.path.join(get_user(kwargs), get_problem(kwargs),
                                                               get_version(kwargs), "MODELS", model_path)))  # unique id
        currentInput["modelFile"] = os.path.join(_globals["MODEL_BUCKET_ADDRESS"], blobs[0].name)

        currentInput["outputDir"] = "gs://{}/{}/{}/{}/RESULTS_STAGING/{}/".format(_globals["MODEL_BUCKET_NAME"],
                                                                                  get_user(kwargs), get_problem(kwargs),
                                                                                  get_version(kwargs),
                                                                                  model_path.split("/")[0])

        S = ScoreJobSpecHandler(algorithm='_'.join(model_path.split("/")[0].split("_")[4:-1]),
                                deployment_config=deployment_config,
                                inputs=currentInput, request_ids={'user': get_user(kwargs),
                                                              'problem': get_problem(kwargs),
                                                              'version': get_version(kwargs)})
        S.create_job_specs()
        T = ScoreJobHandler(deployment_config=deployment_config, job_executor='mlapi')
        T.submit_job(S.job_specs)
        if T.success:
            submitted_scoring_jobs[S.job_specs['jobId']] = model_path.split("/")[0]  # corresponding train job id
            logging.info("Score request successful: {}".format(S.job_specs['jobId']))
        else:
            raise ValueError("Unable to submit score job.")

    # Retrieve scoring
    status = poll(deployment_config, TIME_INTERVAL, submitted_scoring_jobs)
    kwargs['task_instance'].xcom_push(key='successful_jobs', value=get_job_assessment(status))


def aggregate_new(deployment_config, **kwargs):
    """
    Aggregate model scoring. Compute is done remotely.

    :param deployment_config:
    :param kwargs:
    :return:
    """

    _globals = get_deployment_config(deployment_config)
    local_dir, local_destination = get_selector(_globals, kwargs)

    with open(local_destination, 'r') as f:
        selector_dict = json.load(f)
    rmtree(local_dir)

    scoreInput = {
        "scoreDir": "gs://{}/{}/{}/{}/RESULTS_STAGING/".format(_globals["MODEL_BUCKET_NAME"],
                                                               get_user(kwargs),
                                                               get_problem(kwargs),
                                                               get_version(kwargs)),
        "outputDir": kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='output_uri'),
        "masterType": 'n1-standard-4',  # TODO: retrieve dynamically
        "scaleTier": "CUSTOM"
    }

    if selector_dict['aggregation'] == 'average':

        S = PostprocessJobSpecHandler(algorithm='aggregator',
                                      deployment_config=deployment_config,
                                      inputs=scoreInput, request_ids={'user': get_user(kwargs),
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

        # Retrieve scoring
        status = poll(deployment_config, TIME_INTERVAL, submitted_postprocess_jobs)
        kwargs['task_instance'].xcom_push(key='successful_jobs', value=get_job_assessment(status))
    else:
        raise NotImplementedError("Only supported aggregation is: 'average'")


def data_evaluation(deployment_config, **kwargs):

    _globals = get_deployment_config(deployment_config)
    data_path = kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='data_uri')
    # data_path = kwargs['data_path']
    # model_dir = "gs://{}/ET/NUMER/TEST/METADATA/".format(_globals["MODEL_BUCKET_NAME"])
    model_dir = "gs://{}/{}/{}/{}/METADATA/".format(_globals["MODEL_BUCKET_NAME"],
                                                    get_user(kwargs), get_problem(kwargs), get_version(kwargs))

    preprocess_input = {'trainFiles': data_path,
                        'modelDir': model_dir,
                        'scaleTier': 'CUSTOM',
                        'masterType': 'n1-standard-4'
                        }

    S = PreprocessJobSpecHandler(deployment_config=deployment_config,
                                 algorithm='data_evaluator',
                                 append_job_id=False,  # ensure you overwrite same destination
                                 inputs=preprocess_input, request_ids={'user': get_user(kwargs),
                                                              'problem': get_problem(kwargs),
                                                              'version': get_version(kwargs)})
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
