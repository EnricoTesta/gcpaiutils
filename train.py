"""

Implements custom class to submit train jobs.

IMPORTANT: appears that currently there is no way of training using an existing tarball in cloud.
Procedure gives the error: FileNotFoundError: [Errno 2] No such file or directory: '/tmp/pip-req-build-_6j89qcg/setup.py'

The only way I successfully packaged a training application at the moment is via gcloud (specifying a local path).
This means that for each training application I must first submit via gcloud to package it, then I can use it indefinitely
for training using the discovery api.

"""
from googleapiclient import discovery, errors
from datetime import datetime as dt
from yaml import safe_load
import google.auth
import logging
import subprocess
import os


JOB_SPECS_GLOBAL_ARGS = ['scaleTier', 'region', 'modelDir']
JOB_SPECS_DEFAULT_ARGS = ['args', 'trainFiles']

# Load globals and defaults
with open('/gcpaiutils/config/deployment.yml', 'r') as stream:
    GLOBALS = safe_load(stream)
with open('/gcpaiutils/config/defaults.yml', 'r') as stream:
    DEFAULTS = safe_load(stream)
with open('/gcpaiutils/config/hypertune.yml', 'r') as stream:
    HYPER = safe_load(stream)


class TrainJobHandler:
    """Builds train request for GCP AI Platform. Requires job specification as produced by JobSpecHandler.

       Args:
           - project_name: GCP project name
           - job_executor: can be either 'gcloud' or 'mlapi'. The former leverages gcloud to submit train job while
                           the latter uses google's discovery api.

        Main usage:
           - submit_train_job(): returns the object. Sends the job request (async) with the specified parameters.
    """
    def __init__(self, project_name=GLOBALS['PROJECT_NAME'], job_executor='gcloud'):
        self._project_name = project_name
        self.job_executor = job_executor
        self.mlapi = None
        self.hypertune = False
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
        prefix = 'export PATH=/home/vagrant/google-cloud-sdk/bin:$PATH && '
        gcloud = 'gcloud beta ai-platform jobs submit training '
        name = job_spec['jobId'] + ' '
        region = '--region ' + job_spec['trainingInput']['region'] + ' '
        image = '--master-image-uri ' + job_spec['trainingInput']['imageUri'][0] + ' '
        scale = '--scale-tier ' + job_spec['trainingInput']['scaleTier'].lower() + ' '
        if job_spec['trainingInput']['scaleTier'].lower() == 'custom':
            master_machine_type = '--master-machine-type ' + job_spec['trainingInput']['masterType'] + ' '
        else:
            master_machine_type = ''
        if self.hypertune:
            hyper = self.hypertune
        else:
            hyper = ''
        pause = '-- '
        modeldir = '--model-dir=' + job_spec['trainingInput']['modelDir'] + ' '
        train_files = '--train-files=' + job_spec['trainingInput']['trainFiles'] + ' '

        # User-defined args. Even position specify argument name. Odd positions argument values.
        user_defined_args = []
        for i, element in enumerate(job_spec['trainingInput']['args']):
            if i % 2 == 0:  # argument name
                user_defined_args.append('--' + element + '=' + str(job_spec['trainingInput']['args'][i+1]))

        submit_cmd = prefix + gcloud + name + region + image + scale + master_machine_type + hyper + pause + \
                     modeldir + train_files + ' '.join(user_defined_args)
        subprocess.run(submit_cmd, shell=True, check=True)
        self.success = True

    def _exe_job_mlapi(self):
        try:
            self.job_request.execute()
            self.success = True
        except errors.HttpError as err:
            logging.error(err._get_reason())
            self.success = False

    def create_job_request(self, job_spec=None):
        if job_spec is None:
            raise ValueError("Must set job_spec to create a train job.")

        job_spec['trainingInput']['masterConfig'] = {'imageUri': job_spec['trainingInput'].pop('imageUri')[0]}
        job_spec['trainingInput']['args'].append(['model-dir', job_spec['trainingInput'].pop('modelDir')])
        job_spec['trainingInput']['args'].append(['train-files', job_spec['trainingInput'].pop('trainFiles')])

        self.success = None  # reset success flag
        self._auth_setup()
        self.mlapi = discovery.build('ml', 'v1', credentials=self._credentials)
        self.job_request = self.mlapi.projects().jobs().create(body=job_spec
                                                               , parent='projects/{}'.format(self._project_name))

    def submit_train_job(self, job_spec):
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

    def __init__(self, algorithm=None, project_name=GLOBALS['PROJECT_NAME'], train_inputs={}, hypertune=False):
        self.algorithm = algorithm
        self._train_inputs = train_inputs
        try:
            self._train_inputs['imageUri'] = GLOBALS['ATOMS'][self.algorithm]
        except KeyError:
            raise ValueError("Unknown algorithm")
        self._project_name = project_name
        self._train_inputs = train_inputs
        self.hypertune = hypertune

        self.job_specs = None

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
        if self._train_inputs['trainFiles']:
            shards = self._train_inputs['trainFiles'].split("/")  # 3 = subject / 4 = problem / 6 = version
            return shards[3].lower() + '_' + shards[4].lower() + '_' + shards[6].lower() + '_' + \
                   year + month + day + hour + minute + second + '_' + \
                   self._train_inputs['imageUri'][0].split("/")[-1].split(":")[1].lower() + '_' + \
                   self._train_inputs['scaleTier'].lower()
        else:
            return 'j_' + year + month + day + hour + minute + second + '_' + \
                self._train_inputs['imageUri'][0].split("/")[-1].split(":")[1].lower() + '_' + \
                self._train_inputs['scaleTier'].lower()

    def create_job_specs(self):

        spec_full_args = JOB_SPECS_GLOBAL_ARGS + JOB_SPECS_DEFAULT_ARGS
        if self.hypertune:
            spec_full_args += ['hyperparameters']

        # Cast defaults if not found
        for item in spec_full_args:
            if item in self._train_inputs:
                continue

            if item in JOB_SPECS_GLOBAL_ARGS:
                self._train_inputs[item] = GLOBALS[item]
            elif item in JOB_SPECS_DEFAULT_ARGS:
                self._train_inputs[item] = DEFAULTS[self.algorithm][item]
            elif item in ['hyperparameters']:
                self._train_inputs[item] = HYPER[self.algorithm]
            else:
                raise NotImplementedError("Unrecognized job spec argument %s" % item)

        # Generate jobId
        job_id = self._generate_job_name()

        self._train_inputs['modelDir'] = self._train_inputs['modelDir'] + job_id + '/'
        self.job_specs = {'jobId': job_id, 'trainingInput': self._train_inputs}


# TODO: switch to discovery API to submit train requests. This way you can add an HyperparameterSpec object to the dict
"""
# Job Request with HPT
export BUCKET_NAME=ml_train_deploy_test
export MODEL_DIR=sklearn_model_$(date +%Y%m%d_%H%M%S)
export REGION=us-central1
export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --config /Numerai_2.0/mlatoms/ht_config.yml \
  -- \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
  --train-files=gs://$BUCKET_NAME/sample_data/sample_data.csv
"""