import googleapiclient.discovery
from yaml import safe_load
from subprocess import check_call
from utils import get_atom_name_from_dir
import os

# Read configuration file
with open(os.getcwd() + "/config/deployment.yml", 'r') as stream:
    DEPLOYMENT_GLOBALS = safe_load(stream)


class DeploymentHandler:

    def __init__(self, project=DEPLOYMENT_GLOBALS["PROJECT_NAME"],
                 credentials_json=DEPLOYMENT_GLOBALS["GOOGLE_APPLICATION_CREDENTIALS_JSON"]):
        self.project = project
        self.credentials_json = credentials_json

        # Set auth variable
        # TODO: do not use environment variables but instead build discovery api with auth file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_json

    def deploy(self, model_name, model_version, model_origin, region=DEPLOYMENT_GLOBALS["region"],
               runtime=str(DEPLOYMENT_GLOBALS["runtimeVersion"]),
               python_version=str(DEPLOYMENT_GLOBALS["pythonVersion"])):
        if model_version is None:
            raise TypeError("Must specify a model version as a string")
        if model_name is not None:
            self.create_model(model_name, region)
        self.create_version(model_name, model_version, model_origin, runtime, python_version)

    @staticmethod
    def create_model(model_name, region):
        cmd = "gcloud ai-platform models create " + model_name + " --regions " + region
        check_call(cmd, shell=True)

    @staticmethod
    def create_version(model_name, model_version, model_origin, runtime, python_version):

        atom_name = get_atom_name_from_dir(model_origin)

        prefix = "gcloud beta ai-platform versions create " + model_version + " "
        model = "--model " + model_name + " "
        origin = "--origin " + model_origin + " "
        runtime_version = "--runtime-version " + runtime + " "
        python_v = "--python-version " + python_version + " "
        package_uris = "--package-uris " + DEPLOYMENT_GLOBALS["coreBucket"] + \
                       DEPLOYMENT_GLOBALS["PREDICTION"][atom_name]["uri"] + " "
        prediction_class = "--prediction-class " + DEPLOYMENT_GLOBALS["PREDICTION"][atom_name]["class"]

        cmd = prefix + model + origin + runtime_version + python_v + package_uris + prediction_class
        check_call(cmd, shell=True)
