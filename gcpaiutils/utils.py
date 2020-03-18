from yaml import safe_load
from jinja2 import Template
from random import choice
from shutil import rmtree
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
    shards = info_path.split(".")[0].split("_")
    idx = shards.index("info")
    return '_'.join(shards[0:idx]) + '/model_' + '_'.join(shards[idx+1:]) + ".pkl"


def get_hardware_config(atom, data_size):
    if atom in ["class_skl_logreg", "class_lda", "class_qda"]:
        if data_size <= 0.1:
            return "n1-standard-4"
        elif data_size <= 1:
            return "n1-standard-8"
        elif data_size <= 3:
            return "n1-highmem-8"
        else:
            raise(ValueError, "Data size not handled: %s GB." % data_size)
    elif atom in ["class_dummy", "aggregator"]:
        if data_size <= 0.5:
            return "n1-standard-4"
        elif data_size <= 1:
            return "n1-standard-8"
        elif data_size <= 3:
            return "n1-highmem-8"
        else:
            raise(ValueError, "Data size not handled: %s GB." % data_size)
    elif atom in ["class_xgb", "class_lgbm"]:
        if data_size <= 0.1:
            return "n1-highmem-4"
        elif data_size <= 3:
            return "n1-highmem-8"
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
    except FileNotFoundError:
        return Credentials.from_service_account_file(_globals['GCP_AI_PLATFORM_SA'])


def get_metadata(_globals, dag_type, kwargs):

    # Define metadata remote location & setup local dir
    metadata_uri = "{}/{}/{}/METADATA/{}/metadata.json".format(get_user(kwargs),
                                                               get_problem(kwargs),
                                                               get_version(kwargs),
                                                               dag_type)
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
                     if blob.name[-1] != "/"]

    local_dir = make_temp_dir(os.getcwd())
    if len(gcs_blob_list) > 1:
        raise ValueError("More than one selection file found in URI.")
    local_destination = os.path.join(local_dir, gcs_blob_list[0].name.split("/")[-1])
    gcs_blob_list[0].download_to_filename(local_destination, client=gcs_client)
    return local_dir, local_destination
