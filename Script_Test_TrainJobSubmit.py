from train import TrainJobHandler, TrainJobSpecHandler
from google.oauth2.service_account import Credentials
from config.constants import GLOBALS

inputs = {'hyperparameters': {'goal': 'MINIMIZE',
                              'hyperparameterMetricTag': "binary_crossentropy",
                              'maxTrials': 6,
                              'maxParallelTrials': 2,
                              'enableTrialEarlyStopping': True,
                              'params':
                                  [{'parameterName': "C",
                                    'type': 'DOUBLE',
                                    'minValue': 0.1,
                                    'maxValue': 10,
                                    'scaleType': 'UNIT_LOG_SCALE'},
                                   {'parameterName': 'penalty',
                                    'type': 'CATEGORICAL',
                                    'categoricalValues': ["l2", "l1"]}]
                              }
          }

# inputs = {'hyperparameters': {'goal': 'MINIMIZE',
#                               'hyperparameterMetricTag': "categorical_crossentropy",
#                               'maxTrials': 4,
#                               'maxParallelTrials': 2,
#                               'enableTrialEarlyStopping': True,
#                               'params':
#                                   [{'parameterName': "n_estimators",
#                                     'type': 'INTEGER',
#                                     'minValue': 1,
#                                     'maxValue': 10},
#                                    {'parameterName': 'max_depth',
#                                     'type': 'INTEGER',
#                                     'minValue': 1,
#                                     'maxValue': 3}]
#                               }
#           }

S = TrainJobSpecHandler(project_id=GLOBALS["PROJECT_ID"], algorithm='class_xgb',
                        hypertune=False,
                        inputs={'trainFiles': '{}DUMMY/SAMPLES/TRAIN_DATA/MAIN/'.format(GLOBALS["CORE_BUCKET_ADDRESS"])})
S.create_job_specs()

T = TrainJobHandler(credentials=Credentials.from_service_account_file(GLOBALS["AI_PLATFORM_SA"]),
                    project_id=GLOBALS["PROJECT_ID"], job_executor='mlapi')
T.submit_job(S.job_specs)
