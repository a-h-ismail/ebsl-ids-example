#!/usr/bin/env python3

# Copyright (C) 2026 Ahmad Ismail
# SPDX-License-Identifier: MIT
import warnings
from utility import *
from deslib.static.oracle import Oracle
from deslib.des.knora_u import KNORAU
from deslib.des.des_mi import DESMI
from deslib.des.des_p import DESP
from deslib.des.des_knn import DESKNN


def warn(*args, **kwargs):
    pass


# Suppressing sklearn warnings
warnings.warn = warn

# EBSL parameters
max_penalty = 0.7
b = 8
trust_restore_speed = 0.34
conflict_threshold = 0.12
id_col = "StationID"

# Tuning parameters
max_bonus = 0.7
bonus_step = 0.1


fields_to_drop = ['SimulationTime', 'YawRateConfidence', 'Altitude', 'MessageID', 'StationType', 'SemiMajorOrientation',
                  'SemiMajorConfidence', 'SemiMinorConfidence', 'HeadingConfidence', 'AltitudeConfidence',
                  'SpeedConfidence', 'DriveDirection', 'VehicleLength', 'VehicleLengthConfidenceIndication',
                  'VehicleWidth', 'CurvatureConfidence', 'CurvatureCalculationMode', 'ProtocolVersion',
                  'RSU0', 'RSU1', 'RSU2', 'RSU3', "StationID"]
separator = "-"*50

print("Reading the dataset...")
tfeatures, tlabels = from_csv("datasets/train_dataset_6_minutes.csv", "flag", fields_to_drop)
vfeatures, vlabels = from_csv("datasets/validate_dataset_6_minutes.csv", "flag", fields_to_drop)

models = {}

print("Individual models performance:")

# Initialization: Load all stored models and show its performance metrics
for name in ("rf", "ada", "hgb", "mlp", "xgb"):
    models[name] = load_model(name)
    print("Model %s:" % name)
    vpredicted = models[name].predict(vfeatures)
    get_and_print_metrics(vlabels, vpredicted)

for col_names in (("rf", "ada", "hgb"), ("mlp", "ada", "xgb"), ("rf", "mlp", "ada", "xgb", "hgb")):
    for des_type in (Oracle, DESKNN, DESMI, DESP, KNORAU):
        print(separator)
        print("Ensemble of", col_names)
        # Create a list of models to initialize the ensemble model
        model_list = [models[name] for name in col_names]
        des_clf = des_type(pool_classifiers=model_list)

        des_clf.fit(vfeatures, vlabels)
        print("Using %s:" % des_type.__name__)
        if des_type == Oracle:
            vpredict_nobonus = des_clf.predict(vfeatures, vlabels)
        else:
            vpredict_nobonus = des_clf.predict(vfeatures)

        metrics = get_metrics(vlabels, vpredict_nobonus)
        print_metrics(metrics)
