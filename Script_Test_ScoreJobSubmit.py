from predict import ScoreJobHandler, ScoreJobSpecHandler
from google.oauth2.service_account import Credentials
from config.constants import GLOBALS


S = ScoreJobSpecHandler(project_id=GLOBALS["PROJECT_ID"], algorithm='class_qda')
S.create_job_specs()

T = ScoreJobHandler(credentials=Credentials.from_service_account_file(GLOBALS["AI_PLATFORM_SA"]),
                    project_id=GLOBALS["PROJECT_ID"], job_executor='mlapi')
T.submit_job(S.job_specs)
