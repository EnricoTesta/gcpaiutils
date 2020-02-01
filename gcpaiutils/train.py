from gcpaiutils.handler import JobHandler, JobSpecHandler


JOB_SPECS_GLOBAL_ARGS = ['scaleTier', 'region', 'modelDir']
JOB_SPECS_DEFAULT_ARGS = ['args', 'trainFiles']


class TrainJobHandler(JobHandler):
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
        self.hypertune = False

    def translate_job_specs(self, job_spec=None):
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

        return job_spec


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
    def __init__(self, deployment_config=None, algorithm=None, inputs={}, append_job_id=True,
                 request_ids=None, hypertune=False):
        super().__init__(deployment_config=deployment_config, algorithm=algorithm, request_ids=request_ids,
                         append_job_id=append_job_id, inputs=inputs)
        try:
            self.inputs["imageUri"] = self._deployment['ATOMS'][self.algorithm][0]
        except KeyError:
            raise ValueError("Unknown algorithm")
        self.hypertune = hypertune

    def create_job_specs(self):

        spec_full_args = JOB_SPECS_GLOBAL_ARGS + JOB_SPECS_DEFAULT_ARGS
        if self.hypertune:
            spec_full_args += ['hyperparameters', 'hypertuneLoss']

        # Cast defaults if not found
        for item in spec_full_args:
            if item in self.inputs:
                continue

            if item in JOB_SPECS_GLOBAL_ARGS:
                if item == "modelDir":
                    self.inputs[item] = self._globals["MODEL_BUCKET_ADDRESS"]
                else:
                    self.inputs[item] = self._globals[item]
            elif item in JOB_SPECS_DEFAULT_ARGS:
                self.inputs[item] = self._defaults[self.algorithm][item]
            elif item == 'hyperparameters':
                self.inputs[item] = self._hyper[self.algorithm]
            elif item == 'hypertuneLoss':
                self.inputs[item] = self.inputs["hyperparameters"]["hyperparameterMetricTag"]
            else:
                raise NotImplementedError("Unrecognized job spec argument %s" % item)

        # Generate jobId
        job_id = self._generate_job_name(prefix='train')
        if self.append_job_id:
            self.inputs['modelDir'] = self.inputs['modelDir'] + '_'.join(job_id.split("_")[1:]) + '/'
        self.job_specs = {'jobId': job_id, 'trainingInput': self.inputs}
