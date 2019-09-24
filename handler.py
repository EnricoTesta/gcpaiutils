from googleapiclient import errors
from yaml import safe_load
import google.auth
import logging
import subprocess
import os


# Load globals and defaults
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + "/config/deployment.yml", 'r') as stream:
    GLOBALS = safe_load(stream)
with open(dir_path + "/config/defaults.yml", 'r') as stream:
    DEFAULTS = safe_load(stream)
with open(dir_path + "/config/hypertune.yml", 'r') as stream:
    HYPER = safe_load(stream)


class JobHandler:
    """Builds request for GCP AI Platform. Requires job specification as produced by JobSpecHandler.

       Args:
           - project_name: GCP project name
           - job_executor: can be either 'gcloud' or 'mlapi'. The former leverages gcloud to submit train job while
                           the latter uses google's discovery api.

        Main usage:
           - submit_job(): returns the object. Sends the job request (async) with the specified parameters.
    """
    def __init__(self, project_name=GLOBALS['PROJECT_NAME'], job_executor='gcloud'):
        self._project_name = project_name
        self.job_executor = job_executor
        self.mlapi = None
        self._credentials = None
        self._project_id = None
        self.job_request = None
        self.success = None

    def _auth_setup(self):
        # TODO: do not use environment variables but instead build discovery api with auth file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GLOBALS['GOOGLE_APPLICATION_CREDENTIALS_JSON']
        self._credentials, self._project_id = google.auth.default()

    def _execute_job_request(self, job_spec):
        if self.job_executor == 'gcloud':
            self._exe_job_gcloud(job_spec)
        elif self.job_executor == 'mlapi':
            self._exe_job_mlapi()
        else:
            raise NotImplementedError

    def _exe_job_gcloud(self, job_spec):
        pass

    def _exe_job_mlapi(self):
        try:
            self.job_request.execute()  # TODO: manage output (jobId, state, ...)
            self.success = True
        except errors.HttpError as err:
            logging.error(err._get_reason())
            self.success = False

    def create_job_request(self, job_spec=None):
        pass

    def submit_job(self, job_spec):
        if self.job_executor == 'mlapi':
            self.create_job_request(job_spec)
        self._execute_job_request(job_spec)


class JobSpecHandler:
    """Builds job specifications to submit to GCP AI Platform. Specifications can be specified via a dictionary.
    If dictionary is not provided the class fetches defaults from a configuration file.

    Args:
        - project_name: GCP project name
        - train_inputs: a dict specifying ()

    Main usage:
        - create_job_specs(): returns the object with the job_specs property properly configured for a GCP AI
          Platform request.

    """

    def __init__(self, algorithm=None, project_name=GLOBALS['PROJECT_NAME'], inputs={}):
        self.algorithm = algorithm
        self.inputs = inputs
        try:
            self.inputs["imageUri"] = GLOBALS['ATOMS'][self.algorithm]
        except KeyError:
            raise ValueError("Unknown algorithm")
        self._project_name = project_name
        self.job_specs = None

    def _generate_job_name(self):
        pass

    def create_job_specs(self):
        pass
