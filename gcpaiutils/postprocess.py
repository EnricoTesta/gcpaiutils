from gcpaiutils.handler import JobHandler, JobSpecHandler


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

    def translate_job_specs(self, job_spec=None):
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

        return job_spec


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

    def __init__(self, deployment_config=None, algorithm=None, inputs={}, append_job_id=True, request_ids=None):
        super().__init__(deployment_config=deployment_config, algorithm=algorithm,
                         append_job_id=append_job_id, inputs=inputs, request_ids=request_ids)
        try:
            self.inputs["imageUri"] = self._deployment['POSTPROCESS'][self.algorithm][0]
        except KeyError:
            raise ValueError("Unknown algorithm")

    def create_job_specs(self):

        spec_full_args = JOB_SPECS_GLOBAL_ARGS + JOB_SPECS_DEFAULT_ARGS

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
            else:
                raise NotImplementedError("Unrecognized job spec argument %s" % item)

        # Generate specs
        self.job_specs = {'jobId': self._generate_job_name(prefix='postprocess'), 'trainingInput': self.inputs}
