from gcpaiutils.train.trainjobs import TrainJobHandler, JobSpecHandler

S = JobSpecHandler(algorithm='class_askl')
S.create_job_specs()

T = TrainJobHandler(job_executor='mlapi')
T.submit_train_job(S.job_specs)
