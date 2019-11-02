from train import TrainJobHandler, TrainJobSpecHandler
from predict import ScoreJobHandler, ScoreJobSpecHandler
from utils import get_model_path_from_info_path
from googleapiclient import discovery
from google.cloud import storage
from subprocess import check_call
from shutil import rmtree
import logging
import google.auth
from time import sleep
import os

logging.getLogger("OperatorsLogger")
logging.getLogger("OperatorsLogger").setLevel(logging.INFO)

CORE_MODEL_BUCKET = "gs://my_core_model_bucket"
TIME_INTERVAL = 60*1

# auth
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/gauth/mlengine_sa_y.json'
credentials, project_id = google.auth.default()
mlapi = discovery.build('ml', 'v1', credentials=credentials, cache_discovery=False)


def poll(time_interval, project_id, jobs):
    still_running = True
    while still_running:
        status = {}
        states = 1
        for job in jobs:
            request = mlapi.projects().jobs().get(name='projects/' + project_id + '/jobs/' + job)
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


def train(atom=None, train_files=None, master_type=None, hyperspace=None, project_id=None, **kwargs):

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

    S = TrainJobSpecHandler(algorithm=atom, inputs=current_train_input,
                            hypertune=hypertune)  # need copy to refresh args
    S.create_job_specs()
    T = TrainJobHandler(job_executor='mlapi')
    T.submit_job(S.job_specs)
    if T.success:
        submitted_jobs.append(S.job_specs['jobId'])
        logging.info("Train request successful: {}".format(S.job_specs['jobId']))
    else:
        raise ValueError("Unable to submit train job.")

    status = poll(TIME_INTERVAL, project_id, submitted_jobs)

    successful_jobs = [job for job, s in status.items() if s == 'SUCCEEDED']
    failed_jobs = [job for job, s in status.items() if s == 'FAILED']
    if len(failed_jobs) >= 1:
        for job in failed_jobs:
            logging.error("Train job failed: {}".format(job))
        raise ValueError("One or more train jobs failed")
    task_instance = kwargs['task_instance']
    task_instance.xcom_push(key='successful_jobs', value=successful_jobs)


def selection(train_task_ids=None, selector_class=None, **kwargs):
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
        cmd = 'gsutil -m cp {}/{}/info_*.csv {}/{}'.format(CORE_MODEL_BUCKET, job.replace("train_", ""), info_dir,
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


def score(selection_task_id=None, score_dir=None, use_proba=None, master_type=None, project_id=None,
          subject=None, problem=None, version=None, **kwargs):
    task_instance = kwargs['task_instance']
    selected_info = task_instance.xcom_pull(task_ids=selection_task_id, key='selected_info')

    scoreInput = {
        "scoreDir": score_dir,
        "useProba": use_proba,
        "masterType": master_type,
        "scaleTier": "CUSTOM"
    }

    submitted_scoring_jobs = {}

    for info in selected_info:

        currentInput = scoreInput.copy()
        model_path = get_model_path_from_info_path(info)

        # Look for model in appropriate blob (temporary - assumes single model in blob)
        storage_client = storage.Client()
        blobs = list(storage_client.list_blobs("my_core_model_bucket", prefix=model_path))  # unique id
        currentInput["modelFile"] = os.path.join(CORE_MODEL_BUCKET, blobs[0].name)
        currentInput["outputDir"] = "gs://my_customers_bucket/{}/{}/RESULTS/{}/{}/".format(subject,
                                                                                           problem, version,
                                                                                           model_path.split("/")[0])

        S = ScoreJobSpecHandler(algorithm='_'.join(model_path.split("/")[0].split("_")[4:-1]), inputs=currentInput)
        S.create_job_specs()
        T = ScoreJobHandler(job_executor='mlapi')
        T.submit_job(S.job_specs)
        if T.success:
            submitted_scoring_jobs[S.job_specs['jobId']] = model_path.split("/")[0]  # corresponding train job id
            logging.info("Score request successful: {}".format(S.job_specs['jobId']))
        else:
            raise ValueError("Unable to submit score job.")

    # Retrieve scoring
    status = poll(TIME_INTERVAL, project_id, submitted_scoring_jobs)
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
        cmd = "gsutil cp gs://my_customers_bucket/{}/{}/RESULTS/{}/{}/results.csv {}/{}.csv".format(subject, problem,
                                                                                                    version,
                                                                                                    train_job,
                                                                                                    pred_dir,
                                                                                                    train_job)
        check_call(cmd, shell=True)

    kwargs['task_instance'].xcom_push(key='pred_dir', value=pred_dir)
