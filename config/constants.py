from yaml import safe_load
from jinja2 import Template
import os

PATH = os.path.abspath(os.path.dirname(__file__))

# Import global GCP deployment settings from main configuration file
with open("/gauth/z_account_deployment.yml", 'r') as f:
    GLOBALS = safe_load(f)

# Import mlatoms deplyment settings
with open('{}/deployment.yml'.format(PATH), 'r') as f:
    data = f.read()
t = Template(data)
DEPLOYMENT = safe_load(t.render(GLOBALS))

with open("{}/defaults.yml".format(PATH), 'r') as stream:
    DEFAULTS = safe_load(stream)

with open("{}/hypertune.yml".format(PATH), 'r') as stream:
    HYPER = safe_load(stream)