import catboost as cb
import numpy as np
from client3 import distance_ss


def train_dots(clients, target, factor=1, _model=None, fold_num=5):

    print("Training target[", target, "] using factor:", factor)

    folds = [[[] for i in range(fold_num)] for j in range(fold_num)]
    clients_num = len(clients)
    model = _model
    if model == None:
        model = cb.CatBoostClassifier(iterations=100, depth=4, learning_rate=0.04,
                                    #custom_loss=['Recall', 'Precision', 'Accuracy'],
                                    #loss_function='Logloss',
                                    random_seed=4242,
                                    use_best_model=True,
                                    #eval_metric='Logloss',
                                    #task_type='GPU',
                                    logging_level='Verbose')

    features = clients[0].dots[factor-1][0].get_cat_features()

    for c in range(len(clients)):
        client = clients[c]
        client.predicted_target[target] = [.0, .0]
        client.predicted_target_probability[target] = .0

    if fold_num > 0:
        sub_fold_num = clients_num // fold_num
        for i in range(fold_num):
            for j in range(fold_num):
                print(i*clients_num//fold_num + 0+j*sub_fold_num//fold_num, i*clients_num//fold_num + (1+j)*sub_fold_num//fold_num)
                folds[i][j] = clients[i*clients_num//fold_num + 0+j*sub_fold_num//fold_num:i*clients_num//fold_num + (1+j)*sub_fold_num//fold_num]

        total_results = []
        for f in range(len(folds)):
            fold_results = []
            for s in range(len(folds[f])):
                train_data = []
                train_labels = [[], []]
                eval_data = []
                eval_labels = [[],[]]
                dot_clients = []
                for c in range(len(folds[f][s])):
                    client = folds[f][s][c]
                    if client.target[target][0] != .0 and client.target[target][1] != .0:
                        for dot in client.dots[factor-1]:
                            data, labels = dot.get_data()
                            eval_data.append(data)
                            eval_labels[0].append(labels[0])
                            eval_labels[1].append(labels[1])
                            dot_clients.append(client)
                for t in range(len(folds[f])):
                    if t == s:
                        continue
                    for c in range(len(folds[f][t])):
                        client = folds[f][t][c]
                        if client.target[target][0] != .0 and client.target[target][1] != .0:
                            for dot in client.dots[factor-1]:
                                data, labels = dot.get_data()
                                train_data.append(data)
                                train_labels[0].append(labels[0])
                                train_labels[1].append(labels[1])
                fold_train_pool = cb.Pool(np.array(train_data), np.array(train_labels[target]), features)
                fold_eval_pool = cb.Pool(np.array(eval_data), np.array(eval_labels[target]), features)
                model.fit(fold_train_pool, eval_set=fold_eval_pool)
                results = model.predict_proba(fold_eval_pool)
                for i in range(len(results)):
                    if dot_clients[i].predicted_target_probability[target] < results[i][1]:
                        dot_clients[i].predicted_target_probability[target] = results[i][1]
                        dot_clients[i].predicted_target[target] = [float(eval_data[i][0]), float(eval_data[i][1])]#demap_coord([eval_data[i][0], eval_data[i][1]])
                fold_matches = 0.0
                for c in range(len(folds[f][s])):
                    client = folds[f][s][c]
                    if client.target[target][0] != .0 and client.target[target][1] != .0 and \
                        client.predicted_target[target][0] != .0 and client.predicted_target[target][1] != .0 and \
                         distance_ss(client.target[target], client.predicted_target[target]) < 0.02:
                        fold_matches += 1
                fold_results.append(fold_matches / len(folds[f][s]))
            print("Train subfold results:", fold_results)
            total = 0.0
            for s in range(len(folds[f])):
                total += fold_results[s]
            total_results.append(total / len(folds[f]))

        print("Train fold results:", total_results)
        res = 0.0
        for i in total_results:
            res += i
        total_results = res / len(folds)

        print("Train total result:", total_results)

    predict_dots(clients, target, model, factor=factor)
            
    return model


def predict_dots(clients, target, model, factor=1):
    """Predict targets for clients using model and specified dot factor"""

    print("Predicting target[", target, "] using factor:", factor)

    features = clients[0].dots[factor-1][0].get_cat_features()

    train_data = []
    train_labels = [[], []]
    dot_clients = []
    dot_dot = []
    for c in range(len(clients)):
        client = clients[c]
        client.predicted_target[target] = [.0, .0]
        client.predicted_target_probability[target] = .0
        client.best_dot[target][factor-1] = None
        client.best_dot_proba[target][factor-1] = .0
        for dot in client.dots[factor-1]:
            data, labels = dot.get_data()
            train_data.append(data)
            train_labels[0].append(labels[0])
            train_labels[1].append(labels[1])
            dot_clients.append(client)
            dot_dot.append(dot)
    train_pool = cb.Pool(np.array(train_data), np.array(train_labels[target]), features)
    results = model.predict_proba(train_pool)
    for i in range(len(results)):
        if dot_clients[i].predicted_target_probability[target] < results[i][1]:
            dot_clients[i].predicted_target_probability[target] = results[i][1]
            dot_clients[i].predicted_target[target] = [float(train_data[i][0]), float(train_data[i][1])]#demap_coord([train_data[i][0], train_data[i][1]])
            dot_clients[i].best_dot[target][factor-1] = dot_dot[i]
            dot_clients[i].best_dot_proba[target][factor-1] = results[i][1]
    fold_matches = 0.0
    for c in range(len(clients)):
        client = clients[c]
        if client.target[target][0] != .0 and client.target[target][1] != .0 and \
            client.predicted_target[target][0] != .0 and client.predicted_target[target][1] != .0 and \
            distance_ss(client.target[target], client.predicted_target[target]) < 0.02:
            fold_matches += 1
    print("Result on all clients:", fold_matches / len(clients))
    print("Feature importance", model.get_feature_importance(train_pool))


def train_selector(clients, target, _model=None, fold_num=5):

    print("Training selector for target[", target, "]")

    folds = [[[] for i in range(fold_num)] for j in range(fold_num)]
    clients_num = len(clients)
    model = _model
    if model == None:
        model = cb.CatBoostClassifier(iterations=100, depth=4, learning_rate=0.04,
                                    #custom_loss=['Recall', 'Precision', 'Accuracy'],
                                    #loss_function='Logloss',
                                    random_seed=4242,
                                    use_best_model=True,
                                    #eval_metric='Logloss',
                                    #task_type='GPU',
                                    logging_level='Verbose')

    features = clients[0].get_cat_features()

    for c in range(len(clients)):
        client = clients[c]
        client.predicted_target[target] = [.0, .0]
        client.predicted_target_probability[target] = .0

    if fold_num > 0:
        sub_fold_num = clients_num // fold_num
        for i in range(fold_num):
            for j in range(fold_num):
                print(i*clients_num//fold_num + 0+j*sub_fold_num//fold_num, i*clients_num//fold_num + (1+j)*sub_fold_num//fold_num)
                folds[i][j] = clients[i*clients_num//fold_num + 0+j*sub_fold_num//fold_num:i*clients_num//fold_num + (1+j)*sub_fold_num//fold_num]

        total_results = []
        for f in range(len(folds)):
            fold_results = []
            for s in range(len(folds[f])):
                train_data = []
                train_labels = [[], []]
                eval_data = []
                eval_labels = [[],[]]
                dot_clients = []
                for c in range(len(folds[f][s])):
                    client = folds[f][s][c]
                    has_ndots = True
                    for fctr in range(1, len(client.dots)):
                        if len(client.dots[fctr]) == 0:
                            has_ndots = has_ndots and False
                    if client.target[target][0] != .0 and client.target[target][1] != .0 and has_ndots:
                        data, labels = client.get_data()
                        eval_data.append(data)
                        eval_labels[0].append(labels[0])
                        eval_labels[1].append(labels[1])
                        dot_clients.append(client)
                for t in range(len(folds[f])):
                    if t == s:
                        continue
                    for c in range(len(folds[f][t])):
                        client = folds[f][t][c]
                        has_ndots = True
                        for fctr in range(1, len(client.dots)):
                            if len(client.dots[fctr]) == 0:
                                has_ndots = has_ndots and False
                        if client.target[target][0] != .0 and client.target[target][1] != .0 and has_ndots:
                            data, labels = client.get_data()
                            train_data.append(data)
                            train_labels[0].append(labels[0])
                            train_labels[1].append(labels[1])
                fold_train_pool = cb.Pool(np.array(train_data), np.array(train_labels[target]), features)
                fold_eval_pool = cb.Pool(np.array(eval_data), np.array(eval_labels[target]), features)
                model.fit(fold_train_pool, eval_set=fold_eval_pool)
                results = model.predict(fold_eval_pool)
                for i in range(len(results)):
                    dot_clients[i].best_model[target] = dot_clients[i].best_model_from_label(target, results[i][0])
                fold_matches = 0.0
                for c in range(len(folds[f][s])):
                    client = folds[f][s][c]
                    if client.target[target][0] != .0 and client.target[target][1] != .0 \
                            and client.best_dot[target][client.best_model[target]].coords[0] != .0 \
                            and client.best_dot[target][client.best_model[target]].coords[1] != .0 \
                            and distance_ss(client.target[target], client.best_dot[target][client.best_model[target]].coords) < 0.02:
                        fold_matches += 1
                fold_results.append(fold_matches / len(folds[f][s]))
            print("Train subfold results:", fold_results)
            total = 0.0
            for s in range(len(folds[f])):
                total += fold_results[s]
            total_results.append(total / len(folds[f]))

        print("Train fold results:", total_results)
        res = 0.0
        for i in total_results:
            res += i
        total_results = res / len(folds)

        print("Train total result:", total_results)

    predict_selector(clients, target, model)
            
    return model


def predict_selector(clients, target, model):
    """Predict using selector for targets for clients using model and specified dot factor"""

    print("Predicting selector for target[", target, "] ")

    features = clients[0].get_cat_features()

    train_data = []
    train_labels = [[], []]
    dot_clients = []
    for c in range(len(clients)):
        client = clients[c]
        client.predicted_target[target] = [.0, .0]
        client.predicted_target_probability[target] = .0
        has_ndots = True
        for fctr in range(1, len(client.dots)):
            if len(client.dots[fctr]) == 0:
                has_ndots = has_ndots and False
        if has_ndots:
            data, labels = client.get_data()
            train_data.append(data)
            train_labels[0].append(labels[0])
            train_labels[1].append(labels[1])
            dot_clients.append(client)
    train_pool = cb.Pool(np.array(train_data), np.array(train_labels[target]), features)
    results = model.predict(train_pool)
    for i in range(len(results)):
        dot_clients[i].best_model[target] = dot_clients[i].best_model_from_label(target, results[i][0])
    fold_matches = 0.0
    for c in range(len(clients)):
        client = clients[c]
        if client.target[target][0] != .0 and client.target[target][1] != .0 and \
            client.best_dot[target][client.best_model[target]].coords[0] != .0 and client.best_dot[target][client.best_model[target]].coords[1] != .0 and \
            distance_ss(client.target[target], client.best_dot[target][client.best_model[target]].coords) < 0.02:
            fold_matches += 1
    print("Result on all clients:", fold_matches / len(clients))
    print("Feature importance", model.get_feature_importance(train_pool))

