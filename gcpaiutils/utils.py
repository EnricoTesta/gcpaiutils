from yaml import safe_load
from jinja2 import Template
from random import choice
from shutil import rmtree
import string
from datetime import datetime as dt
from google.oauth2.service_account import Credentials
import os


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
    data_size_in_gb = data_size/pow(10, 9)
    if atom in ["class_skl_logreg", "class_lda", "class_qda"]:
        if data_size_in_gb <= 0.1:
            return "n1-standard-4"
        elif data_size_in_gb <= 1:
            return "n1-standard-8"
        else:
            raise(ValueError, "Data size not handled: %s MB." % data_size_in_gb)
    elif atom in ["class_dummy"]:
        if data_size_in_gb <= 0.5:
            return "n1-standard-4"
        elif data_size_in_gb <= 1:
            return "n1-standard-8"
        else:
            raise(ValueError, "Data size not handled: %s MB." % data_size_in_gb)
    elif atom in ["class_xgb", "class_lgbm"]:
        if data_size_in_gb <= 0.1:
            return "n1-highmem-4"
        elif data_size_in_gb <= 1:
            return "n1-highmem-8"
        else:
            raise (ValueError, "Data size not handled: %s MB." % data_size_in_gb)
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


def make_temp_dir(root):

    # Current timestamp
    n = dt.now()
    year = str(n.year)
    month = str(n.month) if n.month > 9 else '0' + str(n.month)
    day = str(n.day) if n.day > 9 else '0' + str(n.day)
    hour = str(n.hour) if n.hour > 9 else '0' + str(n.hour)
    minute = str(n.minute) if n.minute > 9 else '0' + str(n.minute)
    second = str(n.second) if n.second > 9 else '0' + str(n.second)

    # Random ID
    r_id = ''.join([choice(string.ascii_letters + string.digits) for _ in range(10)])

    tmp_dir = "{}/tmp_{}_{}".format(root, ''.join([year, month, day, hour, minute, second]), r_id)
    if os.path.exists(tmp_dir):
        rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    return tmp_dir
