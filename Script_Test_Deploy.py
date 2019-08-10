from deploy import DeploymentHandler

ORIGIN = "gs://my_core_model_bucket/j20190810141341_class_skl_logreg_basic/"

d = DeploymentHandler()
d.deploy(model_name="TestLogReg", model_version="V1", model_origin=ORIGIN)
