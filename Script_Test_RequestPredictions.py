from predict import PredictionHandler


MODEL = "XGB_Final"
CREDENTIALS_PATH = "/gcpaiutils/prediction_key_y.json"

sample_inference_data = {"instances": [[0.62968,0.47129,0.34326,0.22883,0.38267],
[0.37251,0.3781,0.27941,0.74632,0.51044],
[0.51107,0.51159,0.47409,0.34654,0.39853],
[0.64932,0.52581,0.36714,0.20317,0.50647],
[0.35676,0.42657,0.40487,0.6314,0.43897],
[0.34905,0.72961,0.42063,0.30325,0.57263],
[0.3259,0.54084,0.62009,0.42064,0.65099],
[0.55276,0.32829,0.50054,0.51339,0.56894]], "probabilities": True}

p = PredictionHandler(model=MODEL, version="V3", credentials_json=CREDENTIALS_PATH)

for i in range(100):
    results = p.predict_json(sample_inference_data)
    print(results)
