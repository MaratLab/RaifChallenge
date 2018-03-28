import math


class Row:
    def __init__(self, terminal_id=None, customer_id=None, amount=None, city=None,
                 country=None, currency=None, mcc=None, transaction_date=None,
                 atm_address=None, atm_address_lat=None, atm_address_lon=None, pos_address=None, pos_address_lat=None, pos_address_lon=None,
                 work_add_lat=None, work_add_lon=None, home_add_lat=None, home_add_lon=None):
        self.terminal_id = terminal_id
        self.customer_id = customer_id
        self.amount = amount
        self.city = city
        self.country = country
        self.currency = currency
        self.mcc = mcc
        self.transaction_date = transaction_date

        if atm_address_lat is not None:
            self.coords = (atm_address_lat, atm_address_lon)
        elif pos_address_lat is not None:
            self.coords = (pos_address_lat, pos_address_lon)
        else:
            self.coords = [.0, .0]

        if work_add_lat is not None and work_add_lon is not None:
            self.work = [work_add_lat, work_add_lon]
        else:
            self.work = [.0, .0]

        if home_add_lat is not None and home_add_lon is not None:
            self.home = [home_add_lat, home_add_lon]
        else:
            self.home = [.0, .0]

    def __str__(self):
        s = ""
        s += "customer_id:" + str(self.customer_id) + " "
        #s += "city:" + str(self.city) + " "
        s += "mcc:" + str(self.mcc) + " "
        #s += "date:" + str(self.transaction_date) + " "
        s += "coords:" + str(self.coords) + " "
        s += "work:" + str(self.work) + " "
        s += "home:" + str(self.home) + " "

        return s


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def get_weekday_from_row(row):
    return row.transaction_date.weekday()

