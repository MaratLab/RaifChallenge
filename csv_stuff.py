import csv
from row import Row
import datetime


def row_object_from_csv(csv_row):
    if 'work_add_lat' in csv_row:
        work_add_lat = coord_from_csv(csv_row['work_add_lat'])
        work_add_lon = coord_from_csv(csv_row['work_add_lon'])
        home_add_lat = coord_from_csv(csv_row['home_add_lat'])
        home_add_lon = coord_from_csv(csv_row['home_add_lon'])
    else:
        work_add_lat = None
        work_add_lon = None
        home_add_lat = None
        home_add_lon = None
    return Row(terminal_id=csv_row['terminal_id'],
               customer_id=csv_row['customer_id'],
               amount=amount_from_csv(csv_row['amount']),
               city=csv_row['city'],
               country=country_from_csv(csv_row['country']),
               currency=country_from_csv(csv_row['currency']),
               mcc=mcc_from_csv(csv_row['mcc']),
               transaction_date=date_from_csv(csv_row['transaction_date']),
               atm_address=None,
               atm_address_lat=coord_from_csv(csv_row['atm_address_lat']),
               atm_address_lon=coord_from_csv(csv_row['atm_address_lon']),
               pos_address=None,
               pos_address_lat=coord_from_csv(csv_row['pos_address_lat']),
               pos_address_lon=coord_from_csv(csv_row['pos_address_lon']),
               work_add_lat=work_add_lat,
               work_add_lon=work_add_lon,
               home_add_lat=home_add_lat,
               home_add_lon=home_add_lon)


def coord_from_csv(csv_coord):
    if csv_coord == '':
        return None
    else:
        return round_float(float(csv_coord))


def date_from_csv(csv_date):
    if csv_date != '':
        return datetime.datetime.strptime(csv_date, '%Y-%m-%d')
    else:
        return None


def mcc_from_csv(csv_mcc):
    if csv_mcc != '':
        return int(csv_mcc.replace(',', ''))
    else:
        return None


def amount_from_csv(csv_amount):
    if csv_amount != '':
        return max(0.0, float(csv_amount))
    else:
        return 0.0


def country_from_csv(csv_country):
    if csv_country != '':
        return csv_country
    else:
        return None


def currency_from_csv(csv_currency):
    if csv_currency != '':
        return csv_currency
    else:
        return None


def round_float(f):
    return round(f, 6)


def save_solution_to_csv(filename, clients):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['customer_id', 'work_add_lat', 'work_add_lon', 'home_add_lat', 'home_add_lon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        if clients is not None:
            for i in range(len(clients)):
                from random import random
                if clients[i].best_dot[1][clients[i].best_model[1]] is None:
                    if clients[i].best_dot[1][0] is None:
                        work = [.0, .0]
                    else:
                        work = clients[i].best_dot[1][0].coords
                else:
                    work = clients[i].best_dot[1][clients[i].best_model[1]].coords

                if clients[i].best_dot[0][clients[i].best_model[0]] is None:
                    if clients[i].best_dot[0][0] is None:
                        home = [.0, .0]
                    else:
                        home = clients[i].best_dot[0][0].coords
                else:
                    home = clients[i].best_dot[0][clients[i].best_model[0]].coords
                # if random() < 0.5:
                #
                # else:
                #     work = [.0, .0]
                #     home = [.0, .0]
                writer.writerow({'customer_id': clients[i].id, 'work_add_lat': work[0], 'work_add_lon': work[1], 'home_add_lat': home[0], 'home_add_lon': home[1]})


def get_rows_from_csv(filename):
    print("Reading ", filename)
    f = open(filename, newline='', encoding="utf8")
    d = csv.DictReader(f)

    rows = []

    for row in d:
        r = row_object_from_csv(row)
        rows.append(r)

    print("Done.")
    return rows
