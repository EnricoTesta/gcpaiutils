from yaml import safe_load
from jinja2 import Template
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
