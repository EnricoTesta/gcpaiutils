from deploy import DeploymentHandler

ORIGIN = "gs://my_core_model_bucket/j20190813041413_class_xgb_basic/"

d = DeploymentHandler()
d.deploy(model_name="XGB_Final", model_version="V3", model_origin=ORIGIN, build_model=False)
