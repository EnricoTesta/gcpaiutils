import googleapiclient.discovery
from yaml import safe_load
import os

# Read configuration file
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + "/config/deployment.yml", 'r') as stream:
    DEPLOYMENT_GLOBALS = safe_load(stream)


class PredictionHandler:

    def __init__(self, project=DEPLOYMENT_GLOBALS["PROJECT_NAME"],
                 credentials_json=DEPLOYMENT_GLOBALS["GOOGLE_APPLICATION_CREDENTIALS_JSON"],
                 model=None, version=None):
        if model is None:
            raise TypeError("Must provide a valid string reference to an mlatom")
        self.project = project
        self.model = model
        self.credentials_json = credentials_json
        self.version = version

        # Set auth variable
        # TODO: do not use environment variables but instead build discovery api with auth file
        try:
            with open(self.credentials_json, 'r') as f:
                pass
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_json
        except FileNotFoundError:
            pass

        self._api = googleapiclient.discovery.build('ml', 'v1')

    def predict_json(self, instances_dict):
        """Send json data to a deployed model for prediction.
        Args:
            project (str): project where the Cloud ML Engine Model is deployed.
            model (str): model name.
            instances_dict ([Mapping[str: Any]]): Keys should be the names of Tensors
                your deployed model expects as inputs. Values should be datatypes
                convertible to Tensors, or (potentially nested) lists of datatypes
                convertible to tensors.
            version: str, version of the model to target.
        Returns:
            Mapping[str: any]: dictionary of prediction results defined by the
                model.
        """
        # Create the ML Engine service object.
        # To authenticate set the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
        name = 'projects/{}/models/{}'.format(self.project, self.model)

        if self.version is not None:
            name += '/versions/{}'.format(self.version)

        response = self._api.projects().predict(
            name=name,
            body=instances_dict
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response['predictions']
