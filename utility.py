#!/usr/bin/env python3

# Copyright (C) 2026 Ahmad Ismail
# SPDX-License-Identifier: MIT

from math import sqrt
from zipfile import ZIP_LZMA
from skops.io import dump, load
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline


def get_metrics(expected, prediction):
    CM = confusion_matrix(expected, prediction)
    TN, FP = CM[0]
    FN, TP = CM[1]
    print("Metrics:")
    print('Accuracy:   %.3g' % ((TP+TN)/(TP+TN+FP+FN)))
    print('Precision:  %.3g' % (TP/(TP+FP)))
    print('Recall:     %.3g' % (TP/(TP+FN)))
    print('F1 score:   %.3g' % (TP/(TP+0.5*(FP+FN))))
    print('ROC-AUC:    %.3g' % roc_auc_score(expected, prediction))
    print('MCC score:  %.3g' % ((TP * TN - FP * FN) / (sqrt(TP + FP) * sqrt(TP + FN) * sqrt(TN + FP) * sqrt(TN + FN))))
    print('\nConfusion Matrix :\n', CM)
    print()


def from_csv(filename, label_name: str, fields_to_drop: tuple | list | str):
    """Reads the dataset from CSV file, then splits the labels from the data, 
    drops the specified fields and replaces missing values with the mean of the corresponding column"""
    samples = pd.read_csv(filename)
    # Drop unwanted fields
    samples.drop(labels=fields_to_drop, axis=1, inplace=True)
    # Convert strings to numbers (assumes all features are numerical)
    samples[label_name] = pd.to_numeric(samples[label_name], errors='coerce')

    # Drop any unlabeled features
    samples.dropna(subset=[label_name], inplace=True)
    # Split the samples and labels
    labels = samples[label_name]
    samples.drop([label_name], axis=1, inplace=True)

    for col_name in samples.columns:
        # Any missing value in the samples is replaced with the mean of the column
        mean = samples[col_name].mean()
        samples.replace({col_name: np.nan}, mean, inplace=True)

    return samples, labels


def store_model(name, model, scaler):
    """Stores model and scaler in file \"name_model\""""
    sk_pipeline = Pipeline(steps=[('preprocessor', scaler), ('classifier', model)])
    dump(sk_pipeline, "models/%s_model.skops" % name, compression=ZIP_LZMA)


def load_model(name):
    """Retrieves model and scaler from the saved file"""
    return load("models/%s_model.skops" % name, trusted=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier',
                                                         'sklearn.neural_network._stochastic_optimizers.AdamOptimizer', 'sklearn._loss.link.Interval', 'sklearn._loss.link.LogitLink',
                                                         'sklearn._loss.loss.HalfBinomialLoss', 'sklearn.ensemble._hist_gradient_boosting.binning._BinMapper',
                                                         'sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor'])
