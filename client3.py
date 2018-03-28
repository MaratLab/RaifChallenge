from csv_stuff import get_rows_from_csv
import pickle
import numpy as np
import math
import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys, os, time
from scipy.spatial import distance as sd
from itertools import combinations
from random import shuffle

DISTANCE_TO_HOME = 0.02
DISTANCE_TO_WORK = 0.02
COORD_PRECISION = 4
MIN_COORD_DELTA = 1.0 / 10**COORD_PRECISION
CLOSEST_DISTANCE = 0.02
NOT_SO_CLOSE_DOTS = 0.04
CLOSEST_IN_TIME = 1 # days
CLOSEST_IN_TIME_PRECISION = 2
MAP_COORDS_ALPHA = DISTANCE_TO_HOME * math.sqrt(2) / 8
ABSOLUTES_PRECISION = 2
NUMBER_OF_CLOSEST_PRECISION = 2
TIME_SAMPLES = 100
TIME_DISTRIBUTION_PRECISION = 2 # log10(TIME_SAMPLES)
AVERAGE_MINIMAL_TIME_PRECISION = 2
NEARBY_HOME_AND_WORK_DISTANCE = 0.02
AVERAGE_AMOUNT_PRECISION = 3
ABSOLUTE_AMOUNT_PRECISION = 2
DISTANCE_RANGES = [0.01, 0.02, 0.03, 0.04]

CLIENT_STATE_NEW = 0
CLIENT_STATE_DOTS = 1
CLIENT_STATE_FEATURES = 2
CLIENT_STATE_HOMES_WORKS = 3
CLIENT_STATE_CONVERTED_FEATURES_BIT = 4
CLIENT_STATE_PAIRS_BIT = 5
CLIENT_STATE_PAIRS_HWS_BIT = 6
CLIENT_STATE_FINAL_BIT = 15

MAX_FACTOR = 5
MAX_FACTOR_NUM = 200  # maximum number of dots with factor > 1

PREDICTOR_MODEL_NUM = MAX_FACTOR


class Dot:
    def __init__(self, coords=None, atm=None, terminal_id=None, city=None, country=None, mcc=None):
        self.coords = [coords[0], coords[1]]
        self.atm = atm
        self.terminal_id = terminal_id
        self.city = city
        self.country = country
        self.mcc = mcc

        self.dates = []
        self.amounts = []

        self.near = [0, 0] # near targets home nad work

        self.count = 1
                                                            # Float precision:                  Values:
        self.absolute_count = 0.0                           # ABSOULUTES_PRECISION;             [.0; inf]   # needs precision calibration
        self.number_of_closest_dots = 0.0                   # ABSOULUTES_PRECISION;             [.0; inf]   # needs precision calibration
        self.average_distance_to_other_dots = 0.0           # ABSOULUTES_PRECISION              [.0; inf]   # needs precision calibration
        self.number_of_not_so_close_dots = [0.0 for i in range(len(DISTANCE_RANGES))]
        self.number_of_closest_dots_in_time = 0.0           # CLOSEST_IN_TIME_PRECISION;        [.0; inf]   # needs precision calibration
        self.number_of_dots_in_same_day = 0

        self.absolute_nearby_count = [0, 0]                 # [[0; 130] [0; 240]]
    
    def add_transaction(self, date=None, amount=None):
        if date is not None:
            self.dates.append(date)
        if amount is not None and amount != 0.0:
            self.amounts.append(amount)
        self.count += 1

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.atm == other.atm \
                   and self.mcc == other.mcc \
                   and distance_ss(self.coords, other.coords) < MIN_COORD_DELTA
        return NotImplemented

    def __str__(self):
        s = ""
        s += str(self.coords) + " "
        s += str(self.mcc) + " "
        s += str(self.count) + " "
        s += str(self.near) + " "
        s += str(self.absolute_nearby_count) + " "
        return s
    
    def get_data(self):
        """Returns two numpy arrays: data, labels"""
        data = []
        label = []
        data.extend([sfl(self.coords[0]), sfl(self.coords[1])])
                     #data.extend([self.coords[0], self.coords[1]])
        data.extend([self.mcc,
                     #self.count,
                     self.absolute_count,
                     self.number_of_closest_dots,
                     self.average_distance_to_other_dots])
        data.extend(self.number_of_not_so_close_dots)
        data.extend([self.number_of_closest_dots_in_time,
                     self.number_of_dots_in_same_day])
        data.extend(self.absolute_nearby_count)

        label.extend(self.near)
        return data, label

    def get_cat_features(self):
        return np.array([i for i in range(0, len(self.get_data()[0]))])


class Ndot:
    def __init__(self, factor=1):
        self.coords = [.0, .0]
        self.mcc = [0 for i in range(factor)]
        self.same_day = 0
        self.counts = [0 for i in range(factor)]
        self.distance = 0  # from middle to each

        self.near = [0, 0]  # near targets: home and work

        self.absolute_nearby_count = [0, 0] # number of nearby targets: homes and works

    def get_data(self):
        """Returns two numpy arrays: data, labels"""
        data = []
        label = []
        data.extend([sfl(self.coords[0]), sfl(self.coords[1])])
        data.extend(self.mcc)
        data.extend(self.counts)
        data.extend([self.same_day, int(self.distance * 100)])
        data.extend(self.absolute_nearby_count)

        label.extend(self.near)
        return data, label

    def get_cat_features(self):
        return np.array([i for i in range(0, len(self.get_data()[0]))])

    def __str__(self):
        s = ""
        s += str(self.coords) + " "
        s += str(self.mcc) + " "
        s += str(self.same_day) + " "
        s += str(self.counts) + " "
        s += str(self.distance) + " "
        s += str(self.near) + " "
        s += str(self.absolute_nearby_count) + " "
        return s


class Client3:
    def __init__(self, cid):
        self.id = cid
        self.dots = [[] for i in range(MAX_FACTOR)]  # unique client dots, pairs, threes, etc
        self.target = [[.0, .0], [.0, .0]]  # [home, work] coordinates
        self.predicted_target = [[.0, .0], [.0, .0]]
        self.predicted_target_probability = [.0, .0]
        
        self.best_dot = [[None for i in range(PREDICTOR_MODEL_NUM)],
                         [None for i in range(PREDICTOR_MODEL_NUM)]]
        self.best_dot_proba = [[.0 for i in range(PREDICTOR_MODEL_NUM)],
                               [.0 for i in range(PREDICTOR_MODEL_NUM)]]
        self.best_model = [0, 0]
        self.best_model_proba = [.0, .0]

        self.state = [CLIENT_STATE_NEW for i in range(MAX_FACTOR)]

    def str_best(self):
        s = ""
        s += str(self.best_dot) + " "
        s += str(self.best_dot_proba) + " "
        s += str(self.best_model) + " "
        s += str(self.best_model_proba) + " "
        return s

    def check_state(self):
        return self.state

    def get_data(self):
        """Returns two numpy arrays: data, labels"""
        data = []
        label = []
        dot_was_right = [[0 for i in range(len(self.dots))], 
                         [0 for i in range(len(self.dots))]]
        for t in range(2):
            for f in range(len(self.dots)):
                if self.target[t][0] != .0 and self.target[t][1] != .0 \
                  and self.best_dot[t][f].coords[0] != .0 and self.best_dot[t][f].coords[1] != .0 \
                  and distance_ss(self.target[t], self.best_dot[t][f].coords) < 0.02:
                    dot_was_right[t][f] = 1
        
        # for t in range(2):
        #     for f in range(len(self.dots)):
        #         data.extend([sfl(self.best_dot[t][f].coords[0]), sfl(self.best_dot[t][f].coords[1])])
        for t in range(2):
            for f in range(len(self.dots)):
                data.extend([int(round(self.best_dot_proba[t][f], 2) * 10**2)])
        
        for t in range(2):
            for f in range(len(self.dots)):
                d, l = self.best_dot[t][f].get_data()
                data.extend(d[-2:])
        
        # for t in range(2):
        #     for f in range(len(self.dots)):
        #         data.extend([dot_was_right[t][f]])

        ll = [0, 0]
        for t in range(2):
            for f in range(len(self.dots)):
                if dot_was_right[t][f] == 1:
                    ll[t] |= 1 << f

        label.extend(ll)
        return data, label

    def best_model_from_label(self, target, _label):
        label = int(_label)
        if label == 0:
            return 0

        for f in range(len(self.best_dot[target])):
            if label == 1 << f:
                return f
        probabilities = []
        for f in range(len(self.best_dot[target])):
            if ((label >> f) & 0x1) == 1:
                probabilities.append(self.best_dot_proba[target][f])
            else:
                probabilities.append(0.0)
        max_proba = 0
        for f in range(len(self.best_dot[target])):
            if probabilities[max_proba] < probabilities[f]:
                max_proba = f
        return max_proba
    
    def get_cat_features(self):
        return np.array([i for i in range(2*len(self.dots), len(self.get_data()[0]))])

    def add_row(self, row):
        if row.coords[0] != .0 and row.coords[1] != .0:
            if self.target[0][0] == .0 and self.target[0][1] == .0 and row.home[0] != .0 and row.home[1] != .0:
                self.target[0] = [row.home[0], row.home[1]]
            if self.target[1][0] == .0 and self.target[1][1] == .0 and row.work[0] != .0 and row.work[1] != .0:
                self.target[1] = [row.work[0], row.work[1]]
            dot = row_to_dot(row)
            if dot is None:
                return
            known_dot = False
            for i in range(len(self.dots[0])):
                if self.dots[0][i] == dot:
                    if len(dot.dates) > 0 and len(dot.amounts) > 0:
                        self.dots[0][i].add_transaction(date=dot.dates[0], amount=dot.amounts[0])
                    elif len(dot.dates) > 0:
                        self.dots[0][i].add_transaction(date=dot.dates[0])
                    known_dot = True
                    break
            if not known_dot:
                self.dots[0].append(dot)

    def set_features(self):
        # absolute counts
        counts = np.array([self.dots[0][i].count for i in range(len(self.dots[0]))])
        total_count = np.sum(counts)
        abs_counts = counts / total_count
        for i in range(len(self.dots[0])):
            self.dots[0][i].absolute_count = int(round(abs_counts[i], ABSOLUTES_PRECISION) * 10**ABSOLUTES_PRECISION)

        # numbers of transactions
        for i in range(len(self.dots[0])):
            self.dots[0][i].number_of_closest_dots = 0
            for d in range(len(DISTANCE_RANGES)):
                self.dots[0][i].number_of_not_so_close_dots[d] = 0
        distances = distance_mm([self.dots[0][i].coords for i in range(len(self.dots[0]))],
                                [self.dots[0][i].coords for i in range(len(self.dots[0]))])
        sum_of_all_distances = np.sum(distances[0, :])
        for i in range(len(self.dots[0])):
            for j in range(len(self.dots[0])):
                if i == j:
                    continue
                if self.dots[0][i].coords is not None and self.dots[0][j].coords is not None:
                    if distances[i][j] < CLOSEST_DISTANCE:
                        self.dots[0][i].number_of_closest_dots += 1
                    if distances[i][j] < DISTANCE_RANGES[0]:
                        self.dots[0][i].number_of_not_so_close_dots[0] += 1
                    for d in range(1, len(DISTANCE_RANGES)):
                        if DISTANCE_RANGES[d-1] <= distances[i][j] < DISTANCE_RANGES[d]:
                            self.dots[0][i].number_of_not_so_close_dots[d] += 1
            val = int(min(round(sum_of_all_distances / self.dots[0][i].count, ABSOLUTES_PRECISION), 2000)/20)
            self.dots[0][i].average_distance_to_other_dots = val

        # time dimension features

        # close in time
        num = [0 for i in range(len(self.dots[0]))]
        for i in range(len(self.dots[0])):
            for j in range(i+1, len(self.dots[0])):
                is_close = False
                for time1 in self.dots[0][i].dates:
                    if is_close:
                        break
                    for time2 in self.dots[0][j].dates:
                        if is_close:
                            break
                        if time1 - time2 < datetime.timedelta(days=CLOSEST_IN_TIME):
                            is_close = True
                if is_close:
                    num[i] += 1
                    num[j] += 1
            val = int(round(1.0 * num[i] / len(self.dots[0]), CLOSEST_IN_TIME_PRECISION) * 10**NUMBER_OF_CLOSEST_PRECISION)
            self.dots[0][i].number_of_closest_dots_in_time = val
                            
        for i in range(len(self.dots[0])):
            self.dots[0][i].number_of_dots_in_same_day = 0
        for i in range(len(self.dots[0])):
            for j in range(len(self.dots[0])):
                if i == j:
                    continue
                match = 0
                for d1 in self.dots[0][i].dates:
                    if match == 1:
                        break
                    for d2 in self.dots[0][j].dates:
                        if d1 == d2:
                            self.dots[0][i].number_of_dots_in_same_day += 1
                            match = 1
                            break
        for i in range(len(self.dots[0])):
            self.dots[0][i].number_of_dots_in_same_day = int(self.dots[0][i].number_of_dots_in_same_day * 100 / len(self.dots[0]))

        self.set_vicinity(factor=1)

        for t in range(2):
            self.best_dot[t][0] = self.dots[0][0]

        self.state[0] = CLIENT_STATE_FEATURES

    def set_targets(self, targets=None, factor=1):
        """Set nearby homes and works. Input targets are non zero."""
        if len(self.dots[factor-1]) == 0:
            return
        ncoords = [dot.coords for dot in self.dots[factor-1]]
        ntargets = [np.array(targets[0]), np.array(targets[1])]
        ndistances = [distance_mm(ntargets[0], ncoords), distance_mm(ntargets[1], ncoords)]
        nnearby = [ndistances[0] < 0.02, ndistances[1] < 0.02]
        for i in range(len(self.dots[factor-1])):
            self.dots[factor-1][i].absolute_nearby_count = [np.sum(nnearby[0][:, i]), np.sum(nnearby[1][:, i])]
        self.state[factor-1] = CLIENT_STATE_HOMES_WORKS

    def set_vicinity(self, factor=1):
        if len(self.dots[factor-1]) == 0:
            return
        dots = [self.dots[factor-1][i].coords for i in range(len(self.dots[factor-1]))]
        near = [None, None]
        for t in range(2):
            near[t] = distance_sm(self.target[t], dots) < DISTANCE_TO_HOME
        for each in self.dots[factor-1]:
            each.near = [0, 0]
        for i in range(len(self.dots[factor-1])):
            for t in range(2):
                if self.target[t][0] != .0 and self.target[t][1] != .0:
                    if near[t][0][i]:
                        self.dots[factor-1][i].near[t] = 1  # TODO: CHECK THIS!!!

    def set_ndots(self, factor=2):
        # unique dots to make combinations
        udots = self.dots[0]
        if factor < 2 or len(udots) < factor:
            return
        # middles of combinations
        ndots = []

        # if there are fewer possible combinations than MAX_FACTOR_NUM we build all possible combinations
        if len(udots)**factor < MAX_FACTOR_NUM:
            indexes = combinations([i for i in range(len(udots))], factor)  # unique dot combination generator
        # else we get MAX_FACTOR_NUM random combinations of unique dots
        else:
            list_of_indexes = [i for i in range(len(udots))]
            indexes = []
            for i in range(MAX_FACTOR_NUM):
                shuffle(list_of_indexes)
                indexes.append(list_of_indexes[:factor])

        set_list = list(indexes)  # list of combinations (example: [[0,1,2], [0,1,3], [1,2,3]] for combinations of 3 dots out of 4 unique dots )
        dot_list = []  # list of combination rows ([[dot[0], dot[1], dot[2]], ...])
        for s in set_list:
            row = []
            for d in s:
                row.append(udots[d])
            dot_list.append(row)
        coords = [[], []]  # [[row of Xs], [row of Ys]] coordinate rows for each coordinate
        for row in dot_list:
            coors = [[], []]
            for dot in row:
                coors[0].append(dot.coords[0])
                coors[1].append(dot.coords[1])
            coords[0].append(coors[0])
            coords[1].append(coors[1])
        
        ncoords = [np.array(coords[0]), np.array(coords[1])]  # numpy arrays of coordinate rows
        ncoords = [np.sum(ncoords[0], axis=1)/factor, np.sum(ncoords[1], axis=1)/factor]  # numpy arrays of middles' coordinates [[Xs], [Ys]]
        mcoords_list = [[ncoords[0][i], ncoords[1][i]] for i in range(len(coords[0]))]  # list of middles' coordinates [[x0,y0], [x1,y1], ...]
        coords_list = [[coords[0][i][0], coords[1][i][0]] for i in range(len(coords[0]))]  # list of coordinates of first dots in combination row
        ndistances = distance_mm(coords_list, mcoords_list)  # distances from middles to first dots in rows
        for r in range(len(dot_list)):
            ndot = Ndot(factor=factor)
            ndot.coords = [round(ncoords[0][r], COORD_PRECISION), round(ncoords[1][r], COORD_PRECISION)]
            ndot.mcc = [dot.mcc for dot in dot_list[r]]
            ndot.mcc.sort()
            
            match = False
            for date in dot_list[r][0].dates:
                if match:
                    break
                for dot in dot_list[r][1:]:
                    if match:
                        break
                    for adate in dot.dates:
                        if date == adate:
                            match = True
                            break
            if match:
                ndot.same_day = 1
            
            ndot.counts = [dot.count for dot in dot_list[r]]
            ndot.distance = int(round(ndistances[r][r], ABSOLUTES_PRECISION) * 2**ABSOLUTES_PRECISION)  #TODO: write checker
            ndots.append(ndot)

        self.dots[factor - 1] = ndots

        self.set_vicinity(factor=factor)

        for t in range(2):
            self.best_dot[t][factor - 1] = self.dots[factor - 1][0]

        self.state[factor-1] = CLIENT_STATE_FEATURES

        return

    def __str__(self):
        s = str(self.id) + " "
        s += "H:" + str(self.target[0]) + " "
        s += "W:" + str(self.target[1]) + " "
        s += "L:" + str([len(self.dots[f]) for f in range(len(self.dots))]) + " "
        s += "pH:" + str(self.predicted_target[0]) + " "
        s += "pW:" + str(self.predicted_target[1]) + " "
        return s


def fround(coord):
    return round(coord, COORD_PRECISION)


def sfl(fl):
    #return ("{0:."+str(COORD_PRECISION)+"f}").format(fl)
    return str(fl)


def row_to_dot(row):
    if row.transaction_date is not None:
        dot = Dot(coords=[fround(row.coords[0]), fround(row.coords[1])],
                  terminal_id=row.terminal_id,
                  city=row.city,
                  country=row.country,
                  mcc=row.mcc)
        dot.add_transaction(date=row.transaction_date,
                            amount=row.amount)
        return dot
    else:
        return None


def map_coord(coords):
    mcoord = [0.0, 0.0]
    if coords[0] < 0:
        mcoord[0] = coords[0] // MAP_COORDS_ALPHA + 1
    else:
        mcoord[0] = coords[0] // MAP_COORDS_ALPHA
    if coords[1] < 0:
        mcoord[1] = coords[1] // MAP_COORDS_ALPHA + 1
    else:
        mcoord[1] = coords[1] // MAP_COORDS_ALPHA
    return mcoord


def demap_coord(coords):
    return [coords[0] * MAP_COORDS_ALPHA, coords[1] * MAP_COORDS_ALPHA]


def client_list_from_rows(rows):
    clients = []
    cl_dict = {}
    num = 1
    for r in range(len(rows)):
        if r == num * 100000:
            print(r, "complete")
            num += 1
        known_client = False
        if rows[r].customer_id in cl_dict:
            clients[cl_dict[rows[r].customer_id]].add_row(rows[r])
            known_client = True
        # for c in range(len(clients)-1, -1, -1):
        #     if clients[c].id == rows[r].customer_id:
        #         clients[c].add_row(rows[r], clients)
        #         known_client = True
        #         break
        if not known_client:
            new_client = Client3(rows[r].customer_id)
            new_client.add_row(rows[r])
            clients.append(new_client)
            cl_dict[rows[r].customer_id] = len(clients) - 1
    for each in clients:
        each.state[0] = CLIENT_STATE_DOTS
    return clients


def plot_client_dots_features(clients):
    """Plots unique dots features for given clients"""
    
    def loc_plot(values, title):
        plt.title(title)
        plt.plot(values[0], values[1], 'ro', markersize=3)
        plt.show()

    absolute_count = [[], []]
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[0])):
            absolute_count[0].append(clients[i].dots[0][j].absolute_count)
    absolute_count[1] = [i for i in range(len(absolute_count[0]))]
    loc_plot(absolute_count, "absolute_count")

    number_of_closest_dots = [[], []]
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[0])):
            number_of_closest_dots[0].append(clients[i].dots[0][j].number_of_closest_dots)
    number_of_closest_dots[1] = [i for i in range(len(number_of_closest_dots[0]))]
    loc_plot(number_of_closest_dots, "number_of_closest_dots")

    number_of_not_so_close_dots = [[[], []] for d in range(len(DISTANCE_RANGES))]
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[0])):
            for d in range(len(clients[i].dots[0][j].number_of_not_so_close_dots)):
                number_of_not_so_close_dots[d][0].append(clients[i].dots[0][j].number_of_not_so_close_dots[d])
    for d in range(len(DISTANCE_RANGES)):
        number_of_not_so_close_dots[d][1] = [i for i in range(len(number_of_not_so_close_dots[d][0]))]
        loc_plot(number_of_not_so_close_dots[d], "number_of_not_so_close_dots " + str(d))

    '''Actual varies from 0 to 7300. 
       Need to try factor of 2000:
        (if > 2000 than 2000) then divide by 2000 with precision 2
       Then convert to int [0,100]. Which is actually just int(min(v,2000)/20)'''
    average_distance_to_other_dots = [[], []]
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[0])):
            average_distance_to_other_dots[0].append(clients[i].dots[0][j].average_distance_to_other_dots)
    average_distance_to_other_dots[1] = [i for i in range(len(average_distance_to_other_dots[0]))]
    loc_plot(average_distance_to_other_dots, "average_distance_to_other_dots")

    # number_by_weekdays = [[[], []] for i in range(7)]
    # for d in range(7):
    #     for i in range(len(clients)):
    #         for j in range(len(clients[i].dots)):
    #             number_by_weekdays[d][0].append(clients[i].dots[j].number_by_weekdays[d])
    #     number_by_weekdays[d][1] = [i for i in range(len(number_by_weekdays[d][0]))]
    #     loc_plot(number_by_weekdays[d], "number_by_weekdays "+str(d))

    '''Good enough precision is 2. Convertion formula: int(v*10**PRECISION)'''
    number_of_closest_dots_in_time = [[], []]
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[0])):
            number_of_closest_dots_in_time[0].append(clients[i].dots[0][j].number_of_closest_dots_in_time)
    number_of_closest_dots_in_time[1] = [i for i in range(len(number_of_closest_dots_in_time[0]))]
    loc_plot(number_of_closest_dots_in_time, "number_of_closest_dots_in_time")

    number_of_dots_in_same_day = [[], []]
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[0])):
            number_of_dots_in_same_day[0].append(clients[i].dots[0][j].number_of_dots_in_same_day)
    number_of_dots_in_same_day[1] = [i for i in range(len(number_of_dots_in_same_day[0]))]
    loc_plot(number_of_dots_in_same_day, "number_of_dots_in_same_day")

    '''Precision can be set to 1. Max minimal time is 240. Shrink to [0,200]. Convertion: int(min(v,200))'''
    # average_minimal_time_to_other_dots = [[], []]
    # for i in range(len(clients)):
    #     for j in range(len(clients[i].dots)):
    #         average_minimal_time_to_other_dots[0].append(clients[i].dots[j].average_minimal_time_to_other_dots)
    # average_minimal_time_to_other_dots[1] = [i for i in range(len(average_minimal_time_to_other_dots[0]))]
    # loc_plot(average_minimal_time_to_other_dots, "average_minimal_time_to_other_dots")

    '''Precision should be set to 2. Convertion: int(v*SAMPLES)'''
    # time_distribution = [[], []]
    # for i in range(len(clients)):
    #     for j in range(len(clients[i].dots)):
    #         time_distribution[0].append(clients[i].dots[j].time_distribution)
    # time_distribution[1] = [i for i in range(len(time_distribution[0]))]
    # loc_plot(time_distribution, "time_distribution")

    # average_amount = [[], []]
    # for i in range(len(clients)):
    #     for j in range(len(clients[i].dots[0])):
    #         average_amount[0].append(clients[i].dots[0][j].average_amount)
    # average_amount[1] = [i for i in range(len(average_amount[0]))]
    # loc_plot(average_amount, "average_amount")

    # absolute_amount = [[], []]
    # for i in range(len(clients)):
    #     for j in range(len(clients[i].dots[0])):
    #         absolute_amount[0].append(clients[i].dots[0][j].absolute_amount)
    # absolute_amount[1] = [i for i in range(len(absolute_amount[0]))]
    # loc_plot(absolute_amount, "absolute_amount")

    absolute_count_nearby = [[[], []], [[], []]]
    with_one_home_nearby = 0
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[0])):
            absolute_count_nearby[0][0].append(clients[i].dots[0][j].absolute_nearby_count)
            absolute_count_nearby[1][0].append(clients[i].dots[0][j].absolute_nearby_count)
            if clients[i].dots[0][j].absolute_nearby_count[0] == 1:
                with_one_home_nearby += 1
    absolute_count_nearby[0][1] = [i for i in range(len(absolute_count_nearby[0][0]))]
    absolute_count_nearby[1][1] = [i for i in range(len(absolute_count_nearby[1][0]))]
    # print("Dots with one home nearby:", with_one_home_nearby) # 1697
    loc_plot(absolute_count_nearby[0], "absolute_count_homes")
    loc_plot(absolute_count_nearby[1], "absolute_count_works")

    absolute_count_nearby = [[[], []], [[], []]]
    with_one_home_nearby = 0
    for i in range(len(clients)):
        for j in range(len(clients[i].dots[1])):
            absolute_count_nearby[0][0].append(clients[i].dots[1][j].absolute_nearby_count)
            absolute_count_nearby[1][0].append(clients[i].dots[1][j].absolute_nearby_count)
            if clients[i].dots[1][j].absolute_nearby_count[0] == 1:
                with_one_home_nearby += 1
    absolute_count_nearby[0][1] = [i for i in range(len(absolute_count_nearby[0][0]))]
    absolute_count_nearby[1][1] = [i for i in range(len(absolute_count_nearby[1][0]))]
    # print("Dots with one home nearby:", with_one_home_nearby) # 1697
    loc_plot(absolute_count_nearby[0], "absolute_count_homes pairs")
    loc_plot(absolute_count_nearby[1], "absolute_count_works pairs")


def plot_all_homes(clients):
    """Plots homes for all given clients"""
    homes = [[], []]
    for each in clients:
        homes[0].append(each.target[0][0])
        homes[1].append(each.target[0][1])
    fig = plt.gcf()
    ax = fig.gca()
    for i in range(len(homes[0])):
        circle = plt.Circle((homes[0][i], homes[1][i]), 0.02, color='r', fill=False)
        ax.add_artist(circle)
    plt.plot(homes[0], homes[1], 'ro', markersize=3)
    plt.show()


def plot_all_works(clients):
    """Plots works for all given clients"""
    works = [[], []]
    for each in clients:
        if each.target[1][0] != .0 and each.target[1][1] != .0:
            works[0].append(each.target[1][0])
            works[1].append(each.target[1][1])
    fig = plt.gcf()
    ax = fig.gca()
    for i in range(len(works[0])):
        circle = plt.Circle((works[0][i], works[1][i]), 0.02, color='r', fill=False)
        ax.add_artist(circle)
    plt.plot(works[0], works[1], 'ro', markersize=3)
    plt.show()


def plot_probabilities(clients):

    def loc_plot(values, title):
        plt.title(title)
        plt.plot(values[0], values[1], 'ro', markersize=3)
        plt.show()

    proba = [[], []]
    matched_proba = [[], []]
    false_proba = [[], []]
    for each in clients:
        proba[0].append(each.predicted_probability[0])
        if distance_ss(each.predicted_target[0], each.target[0]) < 0.02:
            matched_proba[0].append(each.predicted_probability[0])
        else:
            false_proba[0].append(each.predicted_probability[0])
    proba[1] = [i for i in range(len(proba[0]))]
    matched_proba[1] = [i for i in range(len(matched_proba[0]))]
    false_proba[1] = [i for i in range(len(false_proba[0]))]

    loc_plot(proba, "probabilities")
    loc_plot(matched_proba, "matched probabilities")
    loc_plot(false_proba, "false probabilities")


def plot_best_dot_probabilities(clients):

    def loc_plot(values, title):
        plt.title(title)
        plt.plot(values[0], values[1], 'ro', markersize=3)
        plt.show()

    best_dot_proba = [[[] for i in range(PREDICTOR_MODEL_NUM)],
                      [[] for i in range(PREDICTOR_MODEL_NUM)]]

    for t in range(2):
        for f in range(MAX_FACTOR):
            for c in range(len(clients)):
                best_dot_proba[t][f].append(clients[c].best_dot_proba[t][f])

    for t in range(2):
        for f in range(MAX_FACTOR):
            loc_plot([best_dot_proba[t][f], range(len(best_dot_proba[t][f]))], "best probability " + str(t) + " " + str(f))


def fuck_row(client, row, client_list):
    # print(client)
    # print("Fuck row:", row)
    return
    # home_match = False
    # work_match = False
    # if client.home is not None and row.home is not None and (distance(client.home, row.home) < 0.02):
    #     home_match = True
    # if client.work is not None and row.work is not None and (distance(client.work, row.work) < 0.02):
    #     work_match = True
    # if client_list is not None:
    #     another = None
    #     for c in range(len(client_list)):
    #         if client_list[c].id == row.customer_id:
    #             if client_list[c].home is not None and row.home is not None and (distance(client_list[c].home, row.home) < 0.02) and \
    #                client_list[c].work is not None and row.work is not None and (distance(client_list[c].work, row.work) < 0.02):
    #                 another = client_list[c]
    #                 break
    #     if another is None:
    #         print("Creating new client")
    #         new_id = 0
    #         for c_id in range(1000):
    #             if client_list[-1].id != c_id:
    #                 new_id = c_id
    #         another = Client2(new_id)
    #         client_list.append(another)
    #     another.add_row(row, client_list)


def set_client_features(client):
    client.set_features()
    return client


def set_client_targets(item):
    factor = item[0]
    client = item[1]
    targets = [item[2], item[3]]
    client.set_targets(targets, factor)
    return client


def set_client_ndots(item):
    factor = item[0]
    client = item[1]
    client.set_ndots(factor)
    return client


def load_clients(clients_pickle_file=None, rows_pickle_file=None, csv_file=None):
    if os.path.isfile(clients_pickle_file):
        clients = load(clients_pickle_file)
    else:
        print("No", clients_pickle_file, "file. Will look for", rows_pickle_file, "file...")
        if os.path.isfile(rows_pickle_file):
            rows = load(rows_pickle_file)
        else:
            print("No", rows_pickle_file, "file. Will look for csv file named:", csv_file)
            if os.path.isfile(csv_file):
                rows = get_rows_from_csv(csv_file)
                dump(rows, rows_pickle_file)
            else:
                print("No", csv_file, "file. Quiting")
                sys.exit(1)
        print("Number of rows:", len(rows))
        clients = client_list_from_rows(rows)
        dump(clients, clients_pickle_file)

    print("Number of clients loaded:", len(clients))
    return clients


def fetch(cls=None, _targets=None, max_factor=MAX_FACTOR, clients_pickle_file=None, rows_pickle_file=None, csv_file=None, parallel=False):
    clients = cls
    if clients is None:
        load_clients(clients_pickle_file, rows_pickle_file, csv_file)

    print("Number of clients fetching:", len(clients))

    if clients[0].state[0] < CLIENT_STATE_FEATURES:
        print("Clients have no unique dots features. Populating with features...")
        start = time.time()
        if parallel:
            mp_pool = Pool()
            results = mp_pool.map(set_client_features, clients)
            mp_pool.close()
            mp_pool.join()
            print("Done in parallel")
            clients = results
        else:
            for each in clients:
                each.set_features()
            print("Done")
        end = time.time()
        print("in", format(end - start, '0.4f'), "seconds.")
        dump(clients, clients_pickle_file)
    print("Unique dots ready.")

    for f in range(2, max_factor+1):
        if clients[0].state[f-1] < CLIENT_STATE_FEATURES:
            print("Clients have no dots with factor", f, "Setting...")
            start = time.time()
            if parallel:
                items = []
                for each in clients:
                    items.append([f, each])
                mp_pool = Pool()
                results = mp_pool.map(set_client_ndots, items)
                mp_pool.close()
                mp_pool.join()
                print("Done in parallel")
                clients = results
            else:
                for each in clients:
                    each.set_ndots(f)
                print("Done")
            end = time.time()
            print("in", format(end - start, '0.4f'), "seconds.")
            dump(clients, clients_pickle_file)
    print("All factor dots are set.")

    if _targets is None:
        targets = [[], []]
        start = time.time()
        for each in clients:
            for t in range(2):
                if each.target[t][0] != .0 and each.target[t][1] != .0:
                    targets[t].append(each.target[t])
    else:
        targets = _targets

    for f in range(0, max_factor):
        if clients[0].state[f] < CLIENT_STATE_HOMES_WORKS:
            print("Clients have no homes and works features for dots[", f, "]. Populating with homes and works...")
            if parallel:
                items = []
                for each in clients:
                    items.append([f+1, each, targets[0], targets[1]])
                mp_pool = Pool()
                results = mp_pool.map(set_client_targets, items)
                mp_pool.close()
                mp_pool.join()
                print("Done in parallel")
                clients = results
            else:
                for each in clients:
                    each.set_targets(targets, f+1)
                print("Done")
            end = time.time()
            print("in", format(end - start, '0.4f'), "seconds.")
            dump(clients, clients_pickle_file)

    return clients, targets


def dump(data, filename):
    print("Dumping data...")
    with open(filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Dump", filename, "done")


def load(filename):
    print("Loading data from", filename)
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    print("Data loaded")
    return data


def distance_ss(a, b):
    """2D distance between two dots with coordinate arrays a and b"""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def distance_sm(a, B):
    """2D distances between dot with coordinate array a and array of dots B"""
    na = [a for i in range(len(B))]
    A = np.array(na)
    return sd.cdist(np.array(A), np.array(B), 'euclidean')

def distance_mm(A, B):
    """2D distances between dots from array A to dots from array B"""
    return sd.cdist(np.array(A), np.array(B), 'euclidean')

'''
from scipy.spatial import distance_matrix
    start = time.time()
    d = distance_matrix(a, b)
    end = time.time()
'''

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def cartesian_product_transpose(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)
    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T
