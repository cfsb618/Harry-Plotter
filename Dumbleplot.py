from sklearn.linear_model import LinearRegression

from matplotlib import ticker

import pandas as pd
from numbers_parser import Document
from os import path as path

import random


# TODO: add data from second table to plot

# TODO: implement support for excel
# TODO: implement regression: exponential
# TODO: implement barchart
# TODO: URGENT: remove dictionary as dataset (see jupyter notebook)(what?)
# TODO: no nan-deletion for manual data selection rn


class DataImport:

    def __init__(self, pathtofile, filename):
        self.pathtofile = pathtofile
        self.filename = filename
        self.ext = None
        self.df = None

    def import_data(self, sheet_nr=None, table_nr=None):
        filepath = path.join(self.pathtofile, self.filename)
        _, self.ext = path.splitext(path.basename(filepath))

        if self.ext == ".numbers":
            doc = Document(filepath)
            sheets = doc.sheets
            tables = sheets[sheet_nr].tables
            data = tables[table_nr].rows(values_only=True)
            self.df = pd.DataFrame(data[1:], columns=data[0])
            #self.df.set_index(list(self.df)[0], inplace=True)
        elif self.ext == ".csv":
            self.df = pd.read_csv(filepath)


class Data:

    def __init__(self, row_or_column):
        self.dataset_properties = {}
        self.ydata = {}  # necessary to use second dict?
        self.row_or_column = row_or_column
        self.data_table = None
        self.xdata = []

    def get_data(self, dataframe, first_cell, last_cell):  # better to get cells from __init__()?
        rfc, cfc = first_cell
        rlc, clc = last_cell
        self.data_table = dataframe.loc[rfc:rlc, cfc:clc]

    def make_dict(self, y_data_idx, ext, first_cell, idx):
        # r: start at the correct row
        # offset: adjust for row start, so ydata always starts with "1", necessary to use jupyter script
        n = 1
        if self.row_or_column == "columns":
            for col in y_data_idx:  # starts at 1, assumes that x-data is stored in column 0
                if idx is True:
                    self.dataset_properties["data"] = self.data_table.iloc[:, col]
                else:
                    self.dataset_properties["data"] = self.data_table.loc[:, col]

                self.dataset_properties["name"] = col
                print("added column:", self.dataset_properties["name"])
                self.ydata[n] = self.dataset_properties.copy()
                n += 1

        elif self.row_or_column == "rows":
            for row in y_data_idx:
                if idx is True:
                    self.dataset_properties["data"] = self.data_table.iloc[row, 1:]
                else:
                    # gets index of input name
                    if ext == ".numbers":
                        row_idx = self.data_table.loc[self.data_table[first_cell[1]] == row].index[0]
                        self.dataset_properties["data"] = self.data_table.iloc[row_idx, 1:]
                        self.dataset_properties["name"] = row
                    else:
                        self.dataset_properties["data"] = self.data_table.loc[row, 1:]
                        self.dataset_properties["name"] = row
                print("added row:", self.dataset_properties["name"])
                self.ydata[n] = self.dataset_properties.copy()
                n += 1

        else:
            raise Exception("Error: typo or wrong input"
                            "data_organised_in_rows_or_columns needs to be set to columns or rows")

    def get_xdata(self, data_in_head, x_data_idx, idx):
        if data_in_head is False:
            if self.row_or_column == "columns":
                # Assumption: x-data is stored in first column
                for col in x_data_idx:
                    if idx is True:
                        self.xdata.append(self.data_table.iloc[:, col])
                    else:
                        self.xdata.append(self.data_table.loc[:, col])
            else:
                # Assumption: x-data is stored in first row
                # first cell contains x-data name or can be empty
                for row in x_data_idx:
                    self.xdata.append(self.data_table.iloc[0, row:])

        else:
            head_list = []
            for col in self.data_table.columns:
                head_list.append(col)
            head_list.pop(0)
            self.xdata = pd.DataFrame(head_list)

    def get_errorbar(self, errorbars):
        n = 1
        if errorbars[0] is not None:
            if self.row_or_column == "columns":
                for col in errorbars:  # starts at 1, assumes that x-data is stored in column 0
                    self.ydata[n]["error"] = self.data_table.loc[:, col]
                    n += 1

            elif self.row_or_column == "rows":
                for row in errorbars:
                    self.ydata[n]["error"] = self.data_table.loc[row, 1:]
                    n += 1
        else:
            for i in range(len(self.ydata)):
                n = i + 1
                self.ydata[n]["error"] = None

    def display_data(self):
        # make it display data after nan-removal
        # print("x-data:")
        # print(self.xdata)
        # print("y-data:")
        for data in self.ydata:
            print("x:", self.ydata[data]["xdata"])
            print(data, ":", self.ydata[data]["data"])

    @staticmethod
    def get_idx_from_values(self, ydata):
        notnan_idx = []
        iterator = iter(ydata.notna())
        for i in range(0, len(ydata)):
            if next(iterator):
                notnan_idx.append(i)
        return notnan_idx

    @staticmethod
    def get_values(self, notnanidx, ydata, xdata):
        x_corresponding = []
        y_corresponding = []
        for i in notnanidx:
            x_corresponding.append(xdata.iloc[i])
            y_corresponding.append(ydata.iloc[i])
        return x_corresponding, y_corresponding

    # checks ydata for nan and assigns a xdata-set to each ydata
    # if nan found, deletes it and the corresponding xdata
    def delete_nans(self):
        for dataset, properties in self.ydata.items():
            ydata = self.ydata[dataset]["data"]

            if ydata.isna().any():
                if len(self.xdata) > 1:
                    xdata = self.xdata[dataset]
                else:
                    xdata = self.xdata[0]
                print("Found NaN value")
                notnanidxlist = self.get_idx_from_values(ydata)
                x_vals, y_vals = self.get_values(notnanidxlist, ydata, xdata)
                properties["xdata"], properties["data"] = x_vals, y_vals
            else:
                properties["xdata"] = self.xdata[0]


class Regression:

    def linreg(self, x, y, dataname=None, do_linreg=False):
        if do_linreg:
            model = LinearRegression().fit(x, y)
            r_sq = model.score(x, y)

            if dataname is not None:
                print(dataname, ":")
            print("R^2:", "{:.3f}".format(r_sq))
            print(f"y = {model.coef_} * x + ({model.intercept_})")

            y_pred = model.predict(x)

            return y_pred, model.coef_, model.intercept_


class Plotter:

    def __init__(self, plot_properties):
        self.pp = plot_properties
        self.pp["xtick_number"] = None
        self.pp["ytick_number"] = None

    def set_limits(self):
        if self.pp["x-range"][0] is not None:
            self.pp["x-range"][0] = self.pp["x-range"][0]
            self.pp["x-range"][1] = self.pp["x-range"][1]
        if self.pp["y-range"][0] is not None:
            self.pp["y-range"][0] = self.pp["y-range"][0]
            self.pp["y-range"][1] = self.pp["y-range"][1]

    def set_major_ticks(self):
        if self.pp["x-ticks"] is not None:
            self.pp["xtick_number"] = ticker.MaxNLocator(self.pp["x-ticks"])
        if self.pp["y-ticks"] is not None:
            self.pp["ytick_number"] = ticker.MaxNLocator(self.pp["y-ticks"])

    def set_lin_log(self):
        if self.pp["xaxis-type"] == "lin":
            xaxistype = "linear"
        elif self.pp["xaxis-type"] == "log":
            xaxistype = "log"
        else:
            raise Exception("Error: xaxis-type needs to be set to lin or log")

        if self.pp["yaxis-type"] == "lin":
            yaxistype = "linear"
        elif self.pp["yaxis-type"] == "log":
            yaxistype = "log"
        else:
            raise Exception("Error: yaxis-type needs to be set to lin or log")

        return xaxistype, yaxistype

    def minor_locator(self):
        if self.pp["xaxis-type"] == "log":
            xminor = None
        elif self.pp["x-minor"] is not None:
            xminor = self.pp["x-minor"]
        else:
            xminor = None

        if self.pp["yaxis-type"] == "log":
            yminor = None
        elif self.pp["y-minor"] is not None:
            yminor = self.pp["y-minor"]
        else:
            yminor = None
        return xminor, yminor

    def cast_magic(self):
        print("Accio plot!")
        print("｡ﾟ.")
        magic = random.randint(1, 6)
        if magic == 1:
            print("(｀･ᴗ･)━☆ﾟ･ﾟ:*❤")
        elif magic == 2:
            print("(੭⁰ᴗ⁰)━☆ﾟ.*")
        elif magic == 3:
            print("(੭⁰‿⁰)━☆ﾟ.*･｡ﾟ * ･ ｡")
        elif magic == 4:
            print("ଘ(੭චᴗච)━☆ﾟ.*･")
        elif magic == 5:
            print("(◞ꈍ∇ꈍ)⊃━☆ﾟ⋆*⋆.")
        else:
            print("(∩*✧ω✧)⊃━☆ﾟ.*･｡ﾟ")
        print("*｡ﾟ")
        print(" . ")
