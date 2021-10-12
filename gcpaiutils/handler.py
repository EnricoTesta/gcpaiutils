from googleapiclient import discovery, errors
from google.oauth2.service_account import Credentials
from gcpaiutils.utils import get_deployment_config, get_deployment_constants, get_defaults,\
    get_hyper, get_timestamp_components
import logging
import abc


class JobHandler:
    """Builds request for GCP AI Platform. Requires job specification as produced by JobSpecHandler.

       Args:
           - deployment_config: string specifying deployment configuration YAML file absolute path
           - job_executor: can be either 'gcloud' or 'mlapi'. The former leverages gcloud to submit train job while
                           the latter uses google's discovery api.

        Main usage:
           - submit_job(): returns the object. Sends the job request (async) with the specified parameters.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, deployment_config, job_executor='mlapi'):

        self._globals = get_deployment_config(deployment_config)
        self._project_id = self._globals['PROJECT_ID']
        try:
            self._credentials = Credentials.from_service_account_file(self._globals['AI_PLATFORM_SA'])
        except:
            try:
                self._credentials = Credentials.from_service_account_file(self._globals['GCP_AI_PLATFORM_SA'])
            except KeyError:
                self._credentials = None
        self.job_executor = job_executor
        self.mlapi = None
        self.job_request = None
        self.success = None

    def _execute_job_request(self):
        if self.job_executor == 'mlapi':
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
        self.success = None  # reset success flag
        self.mlapi = discovery.build('ml', 'v1', credentials=self._credentials)
        self.job_request = self.mlapi.projects().jobs().create(body=self.translate_job_specs(job_spec)
                                                               , parent='projects/{}'.format(self._project_id))

    @abc.abstractmethod
    def translate_job_specs(self, job_spec=None):
        pass

    def submit_job(self, job_spec):
        if self.job_executor == 'mlapi':
            self.create_job_request(job_spec)
        self._execute_job_request()


class JobSpecHandler:
    """Builds job specifications to submit to GCP AI Platform. Specifications can be specified via a dictionary.
    If dictionary is not provided the class fetches defaults from a configuration file.

    Args:
        - deployment_config: string specifying deployment configuration YAML file absolute path
        - train_inputs: a dict specifying ()

    Main usage:
        - create_job_specs(): returns the object with the job_specs property properly configured for a GCP AI
          Platform request.

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, deployment_config, algorithm=None, inputs={}, append_job_id=True, request_ids=None):
        self._globals = get_deployment_config(deployment_config)
        self._deployment = get_deployment_constants(self._globals)
        self._defaults = get_defaults()
        self._hyper = get_hyper()
        self.algorithm = algorithm
        self.inputs = inputs
        self.append_job_id = append_job_id
        self.request_ids = request_ids
        self._project_id = self._globals['PROJECT_ID']
        self.job_specs = None

    def _generate_job_name(self, prefix=''):

        year, month, day, hour, minute, second = get_timestamp_components()

        # job name must start with a letter and string must be lowercase
        try:
            return prefix + '_' + self.request_ids['user'].lower().replace("-", "") + '_' + \
                   self.request_ids['problem'].lower().replace("-", "") + '_' + \
                   self.request_ids['version'].lower().replace("-", "") + '_' + \
                   year + month + day + hour + minute + second + '_' + self.algorithm
        except KeyError:
            return 'anonymous' + '_' + prefix + year + month + day + hour + minute + second + '_' + self.algorithm

    @abc.abstractmethod
    def create_job_specs(self):
        pass
