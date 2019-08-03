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


# Load global variables from configuration file
with open('config.yml', 'r') as stream:
    GLOBALS = safe_load(stream)


class TrainJobHandler:

    def __init__(self, project_name=GLOBALS['PROJECT_NAME'], job_executor='gcloud'):
        self._project_name = project_name
        self.job_executor = job_executor
        self.mlapi = None
        self._credentials = None
        self._project_id = None
        self.job_request = None
        self.success = None

    def _auth_setup(self):
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
        pause = '-- '
        modeldir = '--model-dir=' + job_spec['trainingInput']['modelDir'] + ' '
        epochs = '--epochs=' + job_spec['trainingInput']['args'][1]
        submit_cmd = prefix + gcloud + name + region + image + pause + modeldir + epochs
        subprocess.run(submit_cmd, shell=True, check=True)

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

    def __init__(self, project_name):
        self._project_name = project_name

        self._train_inputs = None
        self.job_specs = None

    def _generate_job_name_dev(self):
        n = dt.now()
        year = str(n.year)
        month = str(n.month) if n.month > 9 else '0' + str(n.month)
        day = str(n.day) if n.day > 9 else '0' + str(n.day)
        hour = str(n.hour) if n.hour > 9 else '0' + str(n.hour)
        minute = str(n.minute) if n.minute > 9 else '0' + str(n.minute)
        second = str(n.second) if n.second > 9 else '0' + str(n.second)

        return 'j' + year + month + day + hour + minute + second

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

        return self._train_inputs['scaleTier'] + '_' +\
               self._train_inputs['modelDir'][0].split("/")[-1].split(".")[0].replace('-', '_') + '_' +\
               str(self._train_inputs['args'][1]).split("/")[-1].split(".")[0] + "_" +\
               year + month + day + hour + minute + second

    def training_inputs_from_yaml(self, file_path):
        """Uploads training_inputs from YAML file and generates ready-to-submit specs."""

        with open(file_path, 'r') as stream:
            self._train_inputs = safe_load(stream)

    def create_job_specs(self, yaml_file_path=None):
        if self._train_inputs is None and yaml_file_path is not None:
            self.training_inputs_from_yaml(yaml_file_path)
        if self._train_inputs is None:
            raise ValueError("At least one of yaml_train_inputs and train_inputs must not be None.")
        self.job_specs = {'jobId': self._generate_job_name(), 'trainingInput': self._train_inputs}
