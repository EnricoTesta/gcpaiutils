from predict import ScoreJobHandler, ScoreJobSpecHandler

S = ScoreJobSpecHandler(algorithm='class_skl_logreg')
S.create_job_specs()

T = ScoreJobHandler(job_executor='mlapi')
T.submit_job(S.job_specs)
