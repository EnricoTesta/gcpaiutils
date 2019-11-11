from preprocess import PreprocessJobHandler, PreprocessJobSpecHandler
from google.oauth2.service_account import Credentials
from config.constants import GLOBALS


S = PreprocessJobSpecHandler(project_id=GLOBALS["PROJECT_ID"], algorithm='encoder_onehot',
                             inputs={'trainFiles': 'gs://my_customers_bucket/ET/KAGGLE/STAGING_DATA/STR2INT/'})
S.create_job_specs()

T = PreprocessJobHandler(credentials=Credentials.from_service_account_file(GLOBALS["AI_PLATFORM_SA"]),
                         project_id=GLOBALS["PROJECT_ID"], job_executor='mlapi')
T.submit_job(S.job_specs)
