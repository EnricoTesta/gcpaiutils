from gcpaiutils.train.trainjobs import TrainJobHandler, JobSpecHandler

S = JobSpecHandler(algorithm='class_skl_logreg')
S.create_job_specs()

T = TrainJobHandler()
T.submit_train_job(S.job_specs)
