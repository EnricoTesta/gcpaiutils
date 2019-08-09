from gcpaiutils.train.trainjobs import TrainJobHandler, JobSpecHandler

S = JobSpecHandler(algorithm='class_skl_logreg', train_inputs={'modelDir': 'gs://my_ml_test_bucket/container_test/'})
S.create_job_specs()

T = TrainJobHandler(job_executor='gcloud')
T.submit_train_job(S.job_specs)
