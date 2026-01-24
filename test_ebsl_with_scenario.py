#!/bin/python3

# Copyright (C) 2026 Ahmad Ismail
# SPDX-License-Identifier: MPL-2.0

import ebsl
from utility import *


def compare_to_others(ensemble_model, tlabels):
    print("Hard voting:")
    predicted_hard = ensemble_model._hard_vote()

    m = get_and_print_metrics(tlabels, predicted_hard)
    hard_vote_mcc = m[5]

    print("Soft voting:")

    predicted_soft = ensemble_model._soft_vote()
    m = get_and_print_metrics(tlabels, predicted_soft)
    soft_vote_mcc = m[5]

    return hard_vote_mcc, soft_vote_mcc


fields_to_drop = ['SimulationTime', 'YawRateConfidence', 'Altitude', 'MessageID', 'StationType', 'SemiMajorOrientation',
                  'SemiMajorConfidence', 'SemiMinorConfidence', 'HeadingConfidence', 'AltitudeConfidence',
                  'SpeedConfidence', 'DriveDirection', 'VehicleLength', 'VehicleLengthConfidenceIndication',
                  'VehicleWidth', 'CurvatureConfidence', 'CurvatureCalculationMode', 'ProtocolVersion',
                  'RSU0', 'RSU1', 'RSU2', 'RSU3']
separator = "-"*50

print("Reading the dataset...")
vfeatures, vlabels = from_csv("datasets/validate_dataset_6_minutes.csv", "flag", fields_to_drop)

models = {}

print("Individual models performance:")
# Drop the "StationID since it is meant for EBSL state tracking only"
vfeatures_no_id = vfeatures.drop("StationID", axis=1)
# Initialization: Load all stored models and show its performance metrics
for name in ("rf", "ada", "hgb", "mlp", "xgb"):
    models[name] = load_model(name)
    print("Model %s:" % name)
    vpredicted = models[name].predict(vfeatures_no_id)
    get_and_print_metrics(vlabels, vpredicted)

for col_names in (("rf", "ada", "hgb"), ("mlp", "ada", "xgb"), ("rf", "mlp", "ada", "xgb", "hgb")):
    for BR_choice in ("prior", "trust"):
        print(separator)
        print("Ensemble of", col_names, "in \"%s\" mode:" % BR_choice)
        ebsl_clf = ebsl.EBSL(base_rate_choice=BR_choice,
                             max_penalty=0.8, b=5, trust_restore_speed=0.25, conflict_threshold=0.15, id_col="StationID")

        # Create a BSL_SM for each model and store them in the ensemble classifier
        for name in col_names:
            ebsl_clf.add_model(ebsl.BSL_SM(models[name], None, name))

        ebsl_clf.trust_from_dataset_mcc(vfeatures, vlabels)

        print("\n* Before applying bonuses:")
        print(ebsl_clf)

        vpredict_nobonus = ebsl_clf.predict(vfeatures, _true_labels=vlabels, _keep_caches=True)

        for i in range(len(ebsl_clf._slmodels)):
            m = ebsl_clf._slmodels[i]
            print("Model %s: CICR_0 = %d/%d = %g, CICR_1 = %d/%d = %g"
                  % (m.name, m.nconflict_TN, m.ncumulative_conflict,
                     m.nconflict_TN / m.ncumulative_conflict, m.pconflict_TP,
                     m.pcumulative_conflict, m.pconflict_TP/m.pcumulative_conflict))

        print("\nEBSL (no bonuses):")
        metrics = get_and_print_metrics(vlabels, vpredict_nobonus)

        no_bonus_validation_mcc = metrics[5]

        print("Running the auto-tuning algorithm:")
        ebsl_clf.auto_tune(vfeatures, vlabels, bonus_step=0.1, over_stepping=False,
                           _show_progress=True, descending_order=True)

        vpredicted_with_bonuses = ebsl_clf.predict(vfeatures, _true_labels=vlabels)
        print("\n* After applying bonuses:")

        for i in range(len(ebsl_clf._slmodels)):
            m = ebsl_clf._slmodels[i]
            print("Model %s: CICR_0 = %d/%d = %g, CICR_1 = %d/%d = %g"
                  % (m.name, m.nconflict_TN, m.ncumulative_conflict,
                     m.nconflict_TN / m.ncumulative_conflict, m.pconflict_TP,
                     m.pcumulative_conflict, m.pconflict_TP/m.pcumulative_conflict))
        print("\nEBSL (with bonuses):")
        metrics = get_metrics(vlabels, vpredicted_with_bonuses)
        print_metrics(metrics)
        with_bonus_validation_mcc = metrics[5]
        hard_vote_mcc, soft_vote_mcc = compare_to_others(ebsl_clf, vlabels)
