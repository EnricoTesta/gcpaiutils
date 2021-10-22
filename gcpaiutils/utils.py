from yaml import safe_load
from jinja2 import Template
from random import choice
from shutil import rmtree
from pandas import read_csv
from tempfile import TemporaryDirectory
from google.cloud import storage
import string
import logging
from datetime import datetime as dt
from google.oauth2.service_account import Credentials
import os
import json


PATH = os.path.abspath(os.path.dirname(__file__))


def get_deployment_config(file_path):
    if not (isinstance(file_path, str)):
        raise TypeError("deployment_config must be a string containing the absolute path to your "
                        "deployment configuration YAML file")
    with open(file_path, 'r') as f:
        v = safe_load(f)
    return v


def get_deployment_constants(deployment_config):
    with open('{}/deployment.yml'.format(PATH), 'r') as f:
        data = f.read()
    t = Template(data)
    return safe_load(t.render(deployment_config))


def get_defaults():
    with open("{}/defaults.yml".format(PATH), 'r') as stream:
        defaults = safe_load(stream)
    return defaults


def get_hyper():
    with open("{}/hypertune.yml".format(PATH), 'r') as stream:
        hyper = safe_load(stream)
    return hyper


def get_atom_name_from_dir(job_dir):
    job_name = job_dir.split("/")[-2]
    name_shards = job_name.split("_")
    return "_".join(name_shards[4:-1])


def get_model_path_from_info_path(info_path):
    info_path = info_path.replace("stratified_", "")  # don't mess up with model URI
    shards = info_path.split(".")[0].split("_")
    idx = shards.index("info")
    return '_'.join(shards[0:idx]) + '/model_' + '_'.join(shards[idx+1:])


def get_hardware_config(atom=None, data_size=None, scoring=False):
    """
    Machine types.
    Characteristics double every time last digit in the name doubles. Persistent disks are priced separately.
    Prices are for standard machine types and zone us-central1 (Iowa). E2 machine-types currently not supported by AI Platform.
    Smallest machine types allowed by AI Platforms are n1-standard-4, n1-highmem-2, n1-highcpu-16.
        - E2: cost-optimized. Small to medium workloads that require at most 16 vCPUs but do not require local SSDs or GPUs are an ideal fit.
            * e2-standard-2: 2 vCPU / 8 GB. Price: ~ 0.068 USD/h
            * e2-highmem-2: 2 vCPU / 16 GB. Price: ~ 0.091 USD/h
            * e2-highcpu-2: 2 vCPU / 2 GB. Price: ~ 0.05 USD/h
        - N1: 1st generation general-purpose. (30%-100% more expensive than E2 machines)
            * n1-standard-1: 1 vCPU / 3.75 GB. Price: ~ 0.048 USD/h
            * n1-highmem-2: 2 vCPU / 13 GB. Price: ~ 0.119 USD/h
            * n1-highcpu-2: 2 vCPU / 1.8 GB. Price: ~ 0.071 USD/h
        - GPU: legacy GPU-accelerated machines.
            * standard_gpu: 8 vCPU / 30 GB / 1 NVIDIA Tesla K80. Price: ~ 0.8300 USD/h

    :param atom:
    :param data_size:
    :return:
    """
    if atom == "cusreg_lgbm" and scoring is True:
        if data_size <= 0.1:
            return "n1-standard-8"
        elif data_size <= 1:
            return "n1-highmem-8"
        elif data_size <= 6:
            return "n1-highmem-8"
        else:
            raise(ValueError, "Data size not handled: %s GB." % data_size)
    elif atom in ["class_skl_logreg", "class_lda", "class_qda"]:
        if data_size <= 0.1:
            return "n1-highmem-2" # "n1-standard-4"
        elif data_size <= 1:
            return "n1-highmem-2" # "n1-standard-8"
        elif data_size <= 3:
            return "n1-standard-8" # "n1-standard-8"
        else:
            raise(ValueError, "Data size not handled: %s GB." % data_size)
    elif atom in ["class_dummy", "reg_dummy", "aggregator"]:
        if data_size <= 0.5:
            return "n1-highmem-2" # "n1-standard-4"
        elif data_size <= 1:
            return "n1-highmem-2" # "n1-standard-8"
        elif data_size <= 3:
            return "n1-standard-8" # "n1-standard-8"
        elif data_size <= 8:
            return "n1-highmem-8"
        else:
            raise(ValueError, "Data size not handled: %s GB." % data_size)
    elif atom in ["class_xgb", "class_lgbm", "class_rf", "cusreg_lgbm"]:
        if data_size <= 0.1:
            return "n1-highmem-2" # "n1-standard-4"
        elif data_size <= 1:
            return "n1-standard-8" # "n1-standard-8"
        elif data_size <= 3:
            return "n1-standard-8"
        elif data_size <= 8:
            return "n1-highmem-16"
        else:
            raise (ValueError, "Data size not handled: %s GB." % data_size)
    elif atom in ["class_ffnn"]:
        if data_size <= 1:
            return "n1-standard-4"
        if data_size <= 3:
            return "n1-standard-8"  # "standard_gpu"
        else:
            raise (ValueError, "Data size not handled: %s GB." % data_size)
    else:
        raise(NotImplementedError, "Unrecognized atom name: %s. Could not choose hardware settings." % atom)


def get_user(kwargs):
    return kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='user')


def get_problem(kwargs):
    return kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='problem')


def get_version(kwargs):
    return kwargs['task_instance'].xcom_pull(task_ids='retrieve_params', key='version')


def get_gcs_credentials(_globals):
    try:
        return Credentials.from_service_account_file(_globals['AI_PLATFORM_SA'])
    except:
        try:
            return Credentials.from_service_account_file(_globals['GCP_AI_PLATFORM_SA'])
        except:
            return None


def get_model_metadata(_globals, kwargs):

    # Define metadata remote location & setup local dir
    model_metadata_uri = f"{get_user(kwargs)}/ACTIVE_MODELS/{get_problem(kwargs)}/"

    with TemporaryDirectory() as tmp_dir:
        gcs_credentials = get_gcs_credentials(_globals)
        gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)
        gcs_bucket = gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"])
        blob_list = storage.list_blobs(bucket=gcs_bucket, prefix=model_metadata_uri)
        trained_model_metadata = {}
        for blob in blob_list:
            if blob.name.startswith('featimp'): # assume all models have featimp
                file_name = f"{tmp_dir}/{blob.name.split('/')[-1]}"
                blob.download_to_filename(file_name, client=gcs_client)
                trained_model_metadata[blob.name.split('/')[-1]] = read_csv(file_name)

    return trained_model_metadata


def get_metadata(_globals, dag_type, kwargs):

    # Define metadata remote location & setup local dir
    metadata_uri = f"{get_user(kwargs)}/{get_problem(kwargs)}/METADATA/metadata.json"

    tmp_metadata_dir = make_temp_dir(os.getcwd())
    metadata_local_filename = os.path.join(tmp_metadata_dir, 'metadata.json')

    # Fetch metadata from GCS
    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)
    blob = storage.Blob(metadata_uri, gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"]))
    blob.download_to_filename(metadata_local_filename, client=gcs_client)

    # Load in memory & clean-up
    with open(metadata_local_filename, 'r') as f:
        metadata_dict = json.load(f)
    rmtree(tmp_metadata_dir)

    return metadata_dict


def get_timestamp_components():

    # Current timestamp
    n = dt.now()
    year = str(n.year)
    month = str(n.month) if n.month > 9 else '0' + str(n.month)
    day = str(n.day) if n.day > 9 else '0' + str(n.day)
    hour = str(n.hour) if n.hour > 9 else '0' + str(n.hour)
    minute = str(n.minute) if n.minute > 9 else '0' + str(n.minute)
    second = str(n.second) if n.second > 9 else '0' + str(n.second)

    return year, month, day, hour, minute, second


def make_temp_dir(root):

    year, month, day, hour, minute, second = get_timestamp_components()

    # Random ID
    r_id = ''.join([choice(string.ascii_letters + string.digits) for _ in range(10)])

    tmp_dir = "{}/tmp_{}_{}".format(root, ''.join([year, month, day, hour, minute, second]), r_id)
    if os.path.exists(tmp_dir):
        rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    return tmp_dir


def get_job_assessment(status):
    successful_jobs = [job for job, s in status.items() if s == 'SUCCEEDED']
    failed_jobs = [job for job, s in status.items() if s == 'FAILED']
    if len(failed_jobs) >= 1:
        for job in failed_jobs:
            logging.error("Job failed: {}".format(job))
        raise ValueError("One or more jobs failed")
    return successful_jobs


def get_selector(_globals, kwargs):

    selector_blob = os.path.join(get_user(kwargs), get_problem(kwargs), get_version(kwargs))
    gcs_credentials = get_gcs_credentials(_globals)
    gcs_client = storage.Client(project=_globals['PROJECT_ID'], credentials=gcs_credentials)
    gcs_bucket = gcs_client.get_bucket(_globals["MODEL_BUCKET_NAME"])
    gcs_blob_list = [blob for blob
                     in list(gcs_bucket.list_blobs(prefix=os.path.join(selector_blob, "SELECTOR")))
                     if blob.name.endswith(".json")]

    local_dir = make_temp_dir(os.getcwd())
    local_destination_list = []
    for blob in gcs_blob_list:  # allow multiple selection strategies
        shards = blob.name.split("/")
        os.makedirs(os.path.join(local_dir, shards[-2]))
        local_destination_list.append(os.path.join(local_dir, shards[-2], shards[-1]))
        blob.download_to_filename(local_destination_list[-1], client=gcs_client)
    return local_dir, local_destination_list
