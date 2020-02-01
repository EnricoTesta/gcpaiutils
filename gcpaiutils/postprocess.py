from googleapiclient import discovery
from datetime import datetime as dt
from gcpaiutils.handler import JobHandler, JobSpecHandler
import subprocess


JOB_SPECS_GLOBAL_ARGS = ['scaleTier', 'region']
JOB_SPECS_DEFAULT_ARGS = ['args', 'scoreDir', 'outputDir']


class PostprocessJobHandler(JobHandler):
    """Builds train request for GCP AI Platform. Requires job specification as produced by JobSpecHandler.

       Args:
           - deployment_config: string specifying deployment configuration YAML file absolute path
           - job_executor: can be either 'gcloud' or 'mlapi'. The former leverages gcloud to submit train job while
                           the latter uses google's discovery api.

        Main usage:
           - submit_job(): returns the object. Sends the job request (async) with the specified parameters.
    """
    def __init__(self, deployment_config, job_executor=None):
        super().__init__(deployment_config, job_executor)

    def create_job_request(self, job_spec=None):
        if job_spec is None:
            raise ValueError("Must set job_spec to create a train job.")

        # Map job_spec information to docker entrypoint kwargs
        job_spec['trainingInput']['masterConfig'] = {'imageUri': job_spec['trainingInput'].pop('imageUri')}
        if job_spec['trainingInput']['args'] is None:
            job_spec['trainingInput']['args'] = []
        else:
            for idx, item in enumerate(job_spec['trainingInput']['args']):
                if idx % 2 == 0:  # prefix argument name with '--' to match docker entrypoint kwargs names
                    job_spec['trainingInput']['args'][idx] = '--' + str(job_spec['trainingInput']['args'][idx])
        job_spec['trainingInput']['args'] += ['--score-dir', job_spec['trainingInput'].pop('scoreDir')]
        job_spec['trainingInput']['args'] += ['--output-dir', job_spec['trainingInput'].pop('outputDir')]

        self.success = None  # reset success flag
        self.mlapi = discovery.build('ml', 'v1', credentials=self._credentials)
        self.job_request = self.mlapi.projects().jobs().create(body=job_spec
                                                               , parent='projects/{}'.format(self._project_id))


class PostprocessJobSpecHandler(JobSpecHandler):
    """Builds job specifications to submit to GCP AI Platform. Specifications can be specified via a dictionary.
    If dictionary is not provided the class fetches defaults from a configuration file.

    Args:
        - project_name: GCP project name
        - train_inputs: a dict specifying ()

    Main usage:
        - create_job_specs(): returns the object with the job_specs property properly configured for a GCP AI
          Platform request.

    """

    def __init__(self, deployment_config=None, algorithm=None, inputs={}, append_job_id=True):
        super().__init__(deployment_config=deployment_config, algorithm=algorithm,
                         append_job_id=append_job_id, inputs=inputs)
        try:
            self.inputs["imageUri"] = self._deployment['POSTPROCESS'][self.algorithm][0]
        except KeyError:
            raise ValueError("Unknown algorithm")
        self._postprocess_inputs = inputs

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
        try:
            shards = self._postprocess_inputs['scoreDir'].split("/")  # 3 = subject / 4 = problem / 6 = version
            return 'score_' + shards[3].lower() + '_' + shards[4].lower() + '_' + shards[6].lower() + '_' + \
                   year + month + day + hour + minute + second + '_' + \
                   self.algorithm + '_' + self._postprocess_inputs['scaleTier'].lower()
        except:
            try:
                return 'score_' + self._postprocess_inputs['user'].lower() + '_' + \
                       self._postprocess_inputs['problem'].lower() + '_' + self._postprocess_inputs['version'].lower() + '_' + \
                       year + month + day + hour + minute + second + '_' + \
                       self.algorithm + '_' + self._postprocess_inputs['scaleTier'].lower()
            except:
                return 'scorejob_' + year + month + day + hour + minute + second + '_' + \
                       self.algorithm + '_' + self._postprocess_inputs['scaleTier'].lower()

    def create_job_specs(self):

        spec_full_args = JOB_SPECS_GLOBAL_ARGS + JOB_SPECS_DEFAULT_ARGS

        # Cast defaults if not found
        for item in spec_full_args:
            if item in self._postprocess_inputs:
                continue

            if item in JOB_SPECS_GLOBAL_ARGS:
                if item == "modelDir":
                    self._postprocess_inputs[item] = self._globals["MODEL_BUCKET_ADDRESS"]
                else:
                    self._postprocess_inputs[item] = self._globals[item]
            elif item in JOB_SPECS_DEFAULT_ARGS:
                self._postprocess_inputs[item] = self._defaults[self.algorithm][item]
            else:
                raise NotImplementedError("Unrecognized job spec argument %s" % item)

        # Generate jobId
        job_id = self._generate_job_name()

        # self._postprocess_inputs['modelDir'] = self._postprocess_inputs['modelDir'] + '_'.join(job_id.split("_")[1:]) + '/'
        self.job_specs = {'jobId': job_id, 'trainingInput': self._postprocess_inputs}
