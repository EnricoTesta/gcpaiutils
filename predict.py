from googleapiclient import discovery
from datetime import datetime as dt
from handler import JobHandler, JobSpecHandler, GLOBALS, DEFAULTS
import subprocess
import os


JOB_SPECS_GLOBAL_ARGS = ['scaleTier', 'region']
JOB_SPECS_DEFAULT_ARGS = ['modelFile', 'scoreDir', 'outputDir', 'useProba']


class ScoreJobHandler(JobHandler):
    """Builds train request for GCP AI Platform. Requires job specification as produced by JobSpecHandler.

       Args:
           - project_name: GCP project name
           - job_executor: can be either 'gcloud' or 'mlapi'. The former leverages gcloud to submit train job while
                           the latter uses google's discovery api.

        Main usage:
           - submit_job(): returns the object. Sends the job request (async) with the specified parameters.
    """
    def __init__(self, project_name=GLOBALS['PROJECT_NAME'], job_executor='gcloud'):
        super().__init__(project_name, job_executor)

    def _exe_job_gcloud(self, job_spec):
        prefix = 'export PATH=/home/vagrant/google-cloud-sdk/bin:$PATH && '
        gcloud = 'gcloud beta ai-platform jobs submit training '
        name = job_spec['jobId'] + ' '
        region = '--region ' + job_spec['trainingInput']['region'] + ' '
        image = '--master-image-uri ' + job_spec['trainingInput']['imageUri'] + ' '
        scale = '--scale-tier ' + job_spec['trainingInput']['scaleTier'].lower() + ' '
        if job_spec['trainingInput']['scaleTier'].lower() == 'custom':
            master_machine_type = '--master-machine-type ' + job_spec['trainingInput']['masterType'] + ' '
        else:
            master_machine_type = ''
        pause = '-- '
        model_file = '--model-file=' + job_spec['trainingInput']['modelFile'] + ' '
        score_dir = '--score-dir=' + job_spec['trainingInput']['scoreDir'] + ' '
        output_dir = '--output-dir=' + job_spec['trainingInput']['outputDir'] + ' '
        use_proba = '--use-proba=' + job_spec['trainingInput']['useProba'] + ' '

        submit_cmd = prefix + gcloud + name + region + image + scale + master_machine_type + pause + \
                     model_file + score_dir + output_dir + use_proba
        subprocess.run(submit_cmd, shell=True, check=True)
        self.success = True

    def create_job_request(self, job_spec=None):
        if job_spec is None:
            raise ValueError("Must set job_spec to create a train job.")

        # Map job_spec information to docker entrypoint kwargs
        job_spec['trainingInput']['masterConfig'] = {'imageUri': job_spec['trainingInput'].pop('imageUri')}
        job_spec['trainingInput']['args'] = []
        job_spec['trainingInput']['args'] += ['--model-file', job_spec['trainingInput'].pop('modelFile')]
        job_spec['trainingInput']['args'] += ['--score-dir', job_spec['trainingInput'].pop('scoreDir')]
        job_spec['trainingInput']['args'] += ['--output-dir', job_spec['trainingInput'].pop('outputDir')]
        job_spec['trainingInput']['args'] += ['--use-proba', job_spec['trainingInput'].pop('useProba')]

        self.success = None  # reset success flag
        self._auth_setup()
        self.mlapi = discovery.build('ml', 'v1', credentials=self._credentials)
        self.job_request = self.mlapi.projects().jobs().create(body=job_spec
                                                               , parent='projects/{}'.format(self._project_name))


class ScoreJobSpecHandler(JobSpecHandler):
    """Builds job specifications to submit to GCP AI Platform. Specifications can be specified via a dictionary.
    If dictionary is not provided the class fetches defaults from a configuration file.

    Args:
        - project_name: GCP project name
        - train_inputs: a dict specifying ()

    Main usage:
        - create_job_specs(): returns the object with the job_specs property properly configured for a GCP AI
          Platform request.

    """

    def __init__(self, algorithm=None, project_name=None, inputs={}):
        super().__init__(algorithm, project_name, inputs)
        try:
            self.inputs["imageUri"] = GLOBALS['SCORING'][self.algorithm][0]
        except KeyError:
            raise ValueError("Unknown algorithm")
        self._score_inputs = inputs

    def _generate_job_name(self):
        """Generates jobId (mixed-case letters, numbers, and underscores only, starting with a letter).
        Include info about hardware (scaleTier), training algo (packageUris), train data (--train-files)."""

        n = dt.now()
        year = str(n.year)
        month = str(n.month) if n.month > 9 else '0' + str(n.month)
        day = str(n.day) if n.day > 9 else '0' + str(n.day)
        hour = str(n.hour) if n.hour > 9 else '0' + str(n.hour)
        minute = str(n.minute) if n.minute > 9 else '0' + str(n.minute)
        second = str(n.second) if n.second > 9 else '0' + str(n.second)

        # job name must start with a letter and string must be lowercase
        if self._score_inputs['scoreDir']:
            shards = self._score_inputs['scoreDir'].split("/")  # 3 = subject / 4 = problem / 6 = version
            return 'score_' + shards[3].lower() + '_' + shards[4].lower() + '_' + shards[6].lower() + '_' + \
                   year + month + day + hour + minute + second + '_' + \
                   self.algorithm + '_' + self._score_inputs['scaleTier'].lower()
        else:
            return 'scorejob_' + year + month + day + hour + minute + second + '_' + \
                self.algorithm + '_' + self._score_inputs['scaleTier'].lower()

    def create_job_specs(self):

        spec_full_args = JOB_SPECS_GLOBAL_ARGS + JOB_SPECS_DEFAULT_ARGS

        # Cast defaults if not found
        for item in spec_full_args:
            if item in self._score_inputs:
                continue

            if item in JOB_SPECS_GLOBAL_ARGS:
                self._score_inputs[item] = GLOBALS[item]
            elif item in JOB_SPECS_DEFAULT_ARGS:
                self._score_inputs[item] = DEFAULTS[self.algorithm][item]
            else:
                raise NotImplementedError("Unrecognized job spec argument %s" % item)

        # Generate jobId
        job_id = self._generate_job_name()

        self.job_specs = {'jobId': job_id, 'trainingInput': self._score_inputs}


class PredictionHandler:

    def __init__(self, project=GLOBALS["PROJECT_NAME"],
                 credentials_json=GLOBALS["GOOGLE_APPLICATION_CREDENTIALS_JSON"],
                 model=None, version=None):
        if model is None:
            raise TypeError("Must provide a valid string reference to an mlatom")
        self.project = project
        self.model = model
        self.credentials_json = credentials_json
        self.version = version

        # Set auth variable
        # TODO: do not use environment variables but instead build discovery api with auth file
        try:
            with open(self.credentials_json, 'r') as f:
                pass
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_json
        except:
            pass

        self._api = discovery.build('ml', 'v1')

    def predict_json(self, instances_dict):
        """Send json data to a deployed model for prediction.
        Args:
            project (str): project where the Cloud ML Engine Model is deployed.
            model (str): model name.
            instances_dict ([Mapping[str: Any]]): Keys should be the names of Tensors
                your deployed model expects as inputs. Values should be datatypes
                convertible to Tensors, or (potentially nested) lists of datatypes
                convertible to tensors.
            version: str, version of the model to target.
        Returns:
            Mapping[str: any]: dictionary of prediction results defined by the
                model.
        """
        # Create the ML Engine service object.
        # To authenticate set the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
        name = 'projects/{}/models/{}'.format(self.project, self.model)

        if self.version is not None:
            name += '/versions/{}'.format(self.version)

        response = self._api.projects().predict(
            name=name,
            body=instances_dict
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response['predictions']
