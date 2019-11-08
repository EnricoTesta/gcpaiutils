from googleapiclient import errors
from google.oauth2.service_account import Credentials
from config.constants import GLOBALS
import logging


class JobHandler:
    """Builds request for GCP AI Platform. Requires job specification as produced by JobSpecHandler.

       Args:
           - project_name: GCP project name
           - job_executor: can be either 'gcloud' or 'mlapi'. The former leverages gcloud to submit train job while
                           the latter uses google's discovery api.

        Main usage:
           - submit_job(): returns the object. Sends the job request (async) with the specified parameters.
    """
    def __init__(self, credentials=Credentials.from_service_account_file(GLOBALS["AI_PLATFORM_SA"]),
                 project_id=GLOBALS['PROJECT_ID'], job_executor='mlapi'):
        self._project_id = project_id
        self.job_executor = job_executor
        self.mlapi = None
        self._credentials = credentials
        self._project_id = None
        self.job_request = None
        self.success = None

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

    def __init__(self, algorithm=None, project_id=GLOBALS['PROJECT_ID'], inputs={}):
        self.algorithm = algorithm
        self.inputs = inputs
        self._project_id = project_id
        self.job_specs = None

    def _generate_job_name(self):
        pass

    def create_job_specs(self):
        pass
