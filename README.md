**README**

Requirements:
- mlatoms library deployed to GCP Container Registry
- GCP AI Platform service account credentials (JSON)

Define classes to:
- Submit custom train job to GCP AI platform
- Submit custom score job to GCP AI platform

These classes will probably suffer fast evolution as Google 
makes new ways (APIs) to train/deploy/version/score.

Use this library to *remotely* interact with GCP. If you want to interact from within GCP 
cloud environment, you can probably skip the authentication phase (i.e. use defaults).

Structure:
- config
    - constants.py: initializes GCP global variables (such as credentials), deployment 
    configurations (i.e. URIs to mlatoms deployed on GCP Container Registry), defaults configurations
    , and hyper-parameter tuning default search space for each atom.
    - defaults.yml: defines default arguments for each atom (mainly used for test purposes)
    - deployment.yml: defines Container Registry's URIs for each atom
    - hypertune.yml: defines default hypertune search space for each atom
- handler.py: defines classes JobHandler and JobSpecHandler. These classes contain core GCP interaction 
functionalities and are subclassed in other modules.
- predict.py: defines specific subclasses to handle scoring.
- preprocess.py: defines specific subclasses to handle preprocessing.
- train.py: defines specific subclasses to handle training.
- utils.py: list of functions of general utility
- wrappers.py: defines python wrappers designed to interact with Apache Airflow 
for training, selection, and scoring.