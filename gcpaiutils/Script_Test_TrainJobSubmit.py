from gcpaiutils.train.trainjobs import TrainJobHandler, JobSpecHandler

S = JobSpecHandler(algorithm='class_skl_logreg', train_inputs={'modelDir': 'gs://ml_train_deploy_test/test_87/'})
S.create_job_specs()

T = TrainJobHandler(job_executor='gcloud')
T.submit_train_job(S.job_specs)
