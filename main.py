from row import *

import catboost as cb
from client2 import *
import os, sys

from predictor import train_dots, train_pairs, predict_dots, predict_pairs, train_selector, predict_selector

from csv_stuff import save_solution_to_csv

TRAIN_CSV               = "train_set.csv"
TRAIN_ROWS_PICKLE       = "train.pickle"
TRAIN_CLIENTS_PICKLE    = "train_clients.pickle"

TEST_CSV               = "test_set.csv"
TEST_ROWS_PICKLE       = "test.pickle"
TEST_CLIENTS_PICKLE    = "test_clients.pickle"

PARALLEL = True

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import client3
import predictor3
import random

MAX_FACTOR = client3.MAX_FACTOR

def main3():
    random.seed(4242)
    clients = client3.load_clients(TRAIN_CLIENTS_PICKLE, TRAIN_ROWS_PICKLE, TRAIN_CSV)
    #clients = client3.load_clients(TRAIN_CLIENTS_PICKLE+"_1000", TRAIN_ROWS_PICKLE, TRAIN_CSV)

    clients, targets = client3.fetch(cls=clients[:5000],
                                     _targets=None,
                                     max_factor=MAX_FACTOR,
                                     clients_pickle_file=TRAIN_CLIENTS_PICKLE,#+"_1000",
                                     rows_pickle_file=TRAIN_ROWS_PICKLE,
                                     csv_file=TRAIN_CSV,
                                     parallel=True)

    #client3.plot_client_dots_features(clients)

    models = [[None for i in range(MAX_FACTOR)] for t in range(2)]
    selector = [None for t in range(2)]

    for t in range(2):
        for i in range(MAX_FACTOR):
            models[t][i] = cb.CatBoostClassifier(iterations=2000, depth=4, learning_rate=0.05,
                                                 # custom_loss=['Recall', 'Precision', 'Accuracy'],
                                                 # loss_function='Logloss',
                                                 random_seed=4242,
                                                 use_best_model=True,
                                                 od_type='Iter',
                                                 od_wait=500,
                                                 # eval_metric='Logloss',
                                                 # task_type='GPU',
                                                 #logging_level='Verbose',
                                                 logging_level='Silent')
        selector[t] = cb.CatBoostClassifier(iterations=1000, depth=4, learning_rate=0.05,
                                            # custom_loss=['Recall', 'Precision', 'Accuracy'],
                                            # loss_function='Logloss',
                                            loss_function='MultiClass',
                                            classes_count=32,
                                            random_seed=4242,
                                            use_best_model=True,
                                            od_type='Iter',
                                            od_wait=500,
                                            # eval_metric='Logloss',
                                            # task_type='GPU',
                                            logging_level='Verbose',
                                            #logging_level='Silent'
                                            )

    for t in range(2):
        for i in range(MAX_FACTOR):
            if os.path.isfile("dots_model_"+str(t)+"_"+str(i)):
                models[t][i].load_model(fname="dots_model_"+str(t)+"_"+str(i))
            else:
                models[t][i] = predictor3.train_dots(clients, _model=models[t][i], target=t, factor=i+1)
                models[t][i].save_model("dots_model_" + str(t) + "_" + str(i), format="cbm")

    for t in range(2):
        for i in range(MAX_FACTOR):
            predictor3.predict_dots(clients, target=t, model=models[t][i], factor=i+1)

    # client3.plot_best_dot_probabilities(clients)
    # sys.exit()

    for t in range(2):
        if os.path.isfile("selector_"+str(t)):
            selector[t].load_model(fname="selector_"+str(t))
        else:
            selector[t] = predictor3.train_selector(clients, _model=selector[t], target=t)

        selector[t].save_model("selector_"+str(t), format="cbm")

    for t in range(2):
        for i in range(MAX_FACTOR):
            predictor3.predict_dots(clients, target=t, model=models[t][i], factor=i + 1)
        predictor3.predict_selector(clients, target=t, model=selector[t])

    fold_matches = 0
    none_count = 0
    for t in range(2):
        for c in range(len(clients)):
            if clients[c].target[t][0] != .0 and clients[c].target[t][1] != .0 \
                    and clients[c].best_dot[t][clients[c].best_model[t]] is not None \
                    and clients[c].best_dot[t][clients[c].best_model[t]].coords[0] != .0 \
                    and clients[c].best_dot[t][clients[c].best_model[t]].coords[1] != .0 \
                    and client3.distance_ss(clients[c].target[t],
                                            clients[c].best_dot[t][clients[c].best_model[t]].coords) < 0.02:
                fold_matches += 1
            elif clients[c].best_dot[t][clients[c].best_model[t]] is None:
                if clients[c].best_dot[t][0] is not None \
                        and clients[c].best_dot[t][0] != .0 \
                        and clients[c].best_dot[t][0] != .0 \
                        and client3.distance_ss(clients[c].target[t],
                                                clients[c].best_dot[t][0].coords) < 0.02:
                    fold_matches += 1
                none_count += 1

    print("Matched:", fold_matches)
    print("None:", none_count)


    # test part
    clients = client3.load_clients(TEST_CLIENTS_PICKLE, TEST_ROWS_PICKLE, TEST_CSV)
    #clients = client3.load_clients(TEST_CLIENTS_PICKLE+"_1000", TEST_ROWS_PICKLE, TEST_CSV)
    clients, t = client3.fetch(clients[:1000],
                               _targets=targets,
                               max_factor=MAX_FACTOR,
                               clients_pickle_file=TEST_CLIENTS_PICKLE,#+"_1000",
                               rows_pickle_file=TEST_ROWS_PICKLE,
                               csv_file=TEST_CSV,
                               parallel=True)
    #
    # client3.plot_client_dots_features(clients)
    # client3.plot_all_works(clients)
    # client3.plot_all_homes(clients)

    for t in range(2):
        for i in range(MAX_FACTOR):
            predictor3.predict_dots(clients, target=t, model=models[t][i], factor=i+1)
        predictor3.predict_selector(clients, target=t, model=selector[t])



    client3.dump(clients, "final_test_clients.pickle")
    save_solution_to_csv("test_solution_last_last", clients)
    return

def load_clients_and_save_solution():
    train_clients = client3.load_clients("all_factors_train_clients.pickle", TRAIN_ROWS_PICKLE, TRAIN_CSV)
    clients = client3.load("final_test_clients.pickle")

    print(clients[0].get_data())
    print(clients[0].str_best())

    models = [[None for i in range(MAX_FACTOR)] for t in range(2)]
    selector = [None for t in range(2)]

    for t in range(2):
        for i in range(MAX_FACTOR):
            models[t][i] = cb.CatBoostClassifier(iterations=2000, depth=4, learning_rate=0.04,
                                                 # custom_loss=['Recall', 'Precision', 'Accuracy'],
                                                 # loss_function='Logloss',
                                                 random_seed=4242,
                                                 use_best_model=True,
                                                 od_type='Iter',
                                                 od_wait=500,
                                                 # eval_metric='Logloss',
                                                 # task_type='GPU',
                                                 logging_level='Verbose')
        selector[t] = cb.CatBoostClassifier(iterations=10, depth=4, learning_rate=0.1,
                                            # custom_loss=['Recall', 'Precision', 'Accuracy'],
                                            # loss_function='Logloss',
                                            loss_function='MultiClass',
                                            classes_count=8,
                                            random_seed=4242,
                                            use_best_model=True,
                                            od_type='Iter',
                                            od_wait=500,
                                            # eval_metric='Logloss',
                                            # task_type='GPU',
                                            logging_level='Verbose')

    for t in range(2):
        for i in range(MAX_FACTOR):
            if os.path.isfile("dots_model_" + str(t) + "_" + str(i)):
                models[t][i].load_model(fname="dots_model_" + str(t) + "_" + str(i))

    for t in range(2):
        if os.path.isfile("selector_" + str(t)):
            selector[t].load_model(fname="selector3_" + str(t))

    for t in range(2):
        for i in range(MAX_FACTOR):
            predictor3.predict_dots(train_clients, target=t, model=models[t][i], factor=i + 1)
        predictor3.predict_selector(train_clients, target=t, model=selector[t])
    #
    # client3.dump(train_clients, "all_factors_train_clients.pickle")

    # for t in range(2):
    #     selector[t] = predictor3.train_selector(train_clients, _model=selector[t], target=t)
    #     selector[t].save_model("selector3_" + str(t), format="cbm")

    for t in range(2):
        for i in range(MAX_FACTOR):
            predictor3.predict_dots(clients, target=t, model=models[t][i], factor=i + 1)
        predictor3.predict_selector(clients, target=t, model=selector[t])

    print(clients[0].get_data())
    print(clients[0].str_best())

    #client3.dump(train_clients, "final_train_clients3.pickle")
    #client3.dump(clients, "final_test_clients3.pickle")

    #save_solution_to_csv("test_solution_last", clients)

    print("==================== ============ ======================")

    for t in range(2):
        for c in range(len(train_clients)):
            train_clients[c].best_model[t] = 0
            # train_clients[c].best_model_proba[t] = train_clients[c].best_dot_proba[t][0]
            # for f in range(1, MAX_FACTOR):
            #     if train_clients[c].best_dot_proba[t][f] < train_clients[c].best_model_proba[t]:
            #         train_clients[c].best_model[t] = f
            #         train_clients[c].best_model_proba[t] = train_clients[c].best_dot_proba[t][f]

    fold_matches = 0
    none_count = 0
    for t in range(2):
        for c in range(len(train_clients)):
            if train_clients[c].target[t][0] != .0 and train_clients[c].target[t][1] != .0 \
                    and train_clients[c].best_dot[t][train_clients[c].best_model[t]] is not None \
                    and train_clients[c].best_dot[t][train_clients[c].best_model[t]].coords[0] != .0 \
                    and train_clients[c].best_dot[t][train_clients[c].best_model[t]].coords[1] != .0 \
                    and client3.distance_ss(train_clients[c].target[t], train_clients[c].best_dot[t][train_clients[c].best_model[t]].coords) < 0.02:
                fold_matches += 1
            elif train_clients[c].best_dot[t][train_clients[c].best_model[t]] is None:
                if train_clients[c].best_dot[t][0] is not None \
                        and train_clients[c].best_dot[t][0] != .0 \
                        and train_clients[c].best_dot[t][0] != .0 \
                        and client3.distance_ss(train_clients[c].target[t], train_clients[c].best_dot[t][0].coords) < 0.02:
                    fold_matches += 1
                none_count += 1

    print("Matched:", fold_matches)
    print("None:", none_count)

    for t in range(2):
        for c in range(len(clients)):
            clients[c].best_model[t] = 0
            # clients[c].best_model_proba[t] = clients[c].best_dot_proba[t][0]
            # for f in range(1, MAX_FACTOR):
            #     if clients[c].best_dot_proba[t][f] < clients[c].best_model_proba[t]:
            #         clients[c].best_model[t] = f
            #         clients[c].best_model_proba[t] = clients[c].best_dot_proba[t][f]

    save_solution_to_csv("test_solution_lasT_last", clients)


if __name__ == "__main__":

    main3()
