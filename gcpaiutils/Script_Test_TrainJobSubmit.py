from gcpaiutils.train.trainjobs import TrainJobHandler, JobSpecHandler
from os import getcwd

PROJECT_NAME = "num-00"

S = JobSpecHandler(PROJECT_NAME)
S.create_job_specs(getcwd() + '/train/default_job_specs_docker.yml')

T = TrainJobHandler(PROJECT_NAME)
T.submit_train_job(S.job_specs)
