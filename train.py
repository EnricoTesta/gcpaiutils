from googleapiclient import discovery
from datetime import datetime as dt
from handler import JobHandler, JobSpecHandler, GLOBALS, DEFAULTS, HYPER
import subprocess


JOB_SPECS_GLOBAL_ARGS = ['scaleTier', 'region', 'modelDir']
JOB_SPECS_DEFAULT_ARGS = ['args', 'trainFiles']


class TrainJobHandler(JobHandler):
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
        self.hypertune = False

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

    def create_job_request(self, job_spec=None):
        if job_spec is None:
            raise ValueError("Must set job_spec to create a train job.")
        if 'hyperparameters' in job_spec['trainingInput']:
            self.hypertune = True

        # Map job_spec information to docker entrypoint kwargs
        job_spec['trainingInput']['masterConfig'] = {'imageUri': job_spec['trainingInput'].pop('imageUri')}
        if job_spec['trainingInput']['args'] is None:
            job_spec['trainingInput']['args'] = []
        else:
            for idx, item in enumerate(job_spec['trainingInput']['args']):
                if idx % 2 == 0:  # prefix argument name with '--' to match docker entrypoint kwargs names
                    job_spec['trainingInput']['args'][idx] = '--' + str(job_spec['trainingInput']['args'][idx])
        job_spec['trainingInput']['args'] += ['--model-dir', job_spec['trainingInput'].pop('modelDir')]
        job_spec['trainingInput']['args'] += ['--train-files', job_spec['trainingInput'].pop('trainFiles')]
        if self.hypertune:
            job_spec['trainingInput']['args'] += ['--hypertune-loss', job_spec['trainingInput'].pop('hypertuneLoss')]

        self.success = None  # reset success flag
        self._auth_setup()
        self.mlapi = discovery.build('ml', 'v1', credentials=self._credentials)
        self.job_request = self.mlapi.projects().jobs().create(body=job_spec
                                                               , parent='projects/{}'.format(self._project_name))


class TrainJobSpecHandler(JobSpecHandler):
    """Builds job specifications to submit to GCP AI Platform. Specifications can be specified via a dictionary.
    If dictionary is not provided the class fetches defaults from a configuration file.

    Args:
        - project_name: GCP project name
        - train_inputs: a dict specifying ()

    Main usage:
        - create_job_specs(): returns the object with the job_specs property properly configured for a GCP AI
          Platform request.

    """

    def __init__(self, algorithm=None, project_name=None, inputs={}, hypertune=False):
        super().__init__(algorithm, project_name, inputs)
        try:
            self.inputs["imageUri"] = GLOBALS['ATOMS'][self.algorithm][0]
        except KeyError:
            raise ValueError("Unknown algorithm")
        self._train_inputs = inputs
        self.hypertune = hypertune

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
            return 'train_' + shards[3].lower() + '_' + shards[4].lower() + '_' + shards[6].lower() + '_' + \
                   year + month + day + hour + minute + second + '_' + \
                   self.algorithm + '_' + self._train_inputs['scaleTier'].lower()
        else:
            return 'trainjob_' + year + month + day + hour + minute + second + '_' + \
                self.algorithm + '_' + self._train_inputs['scaleTier'].lower()

    def create_job_specs(self):

        spec_full_args = JOB_SPECS_GLOBAL_ARGS + JOB_SPECS_DEFAULT_ARGS
        if self.hypertune:
            spec_full_args += ['hyperparameters', 'hypertuneLoss']

        # Cast defaults if not found
        for item in spec_full_args:
            if item in self._train_inputs:
                continue

            if item in JOB_SPECS_GLOBAL_ARGS:
                self._train_inputs[item] = GLOBALS[item]
            elif item in JOB_SPECS_DEFAULT_ARGS:
                self._train_inputs[item] = DEFAULTS[self.algorithm][item]
            elif item == 'hyperparameters':
                self._train_inputs[item] = HYPER[self.algorithm]
            elif item == 'hypertuneLoss':
                self._train_inputs[item] = self._train_inputs["hyperparameters"]["hyperparameterMetricTag"]
            else:
                raise NotImplementedError("Unrecognized job spec argument %s" % item)

        # Generate jobId
        job_id = self._generate_job_name()

        self._train_inputs['modelDir'] = self._train_inputs['modelDir'] + '_'.join(job_id.split("_")[1:]) + '/'
        self.job_specs = {'jobId': job_id, 'trainingInput': self._train_inputs}
