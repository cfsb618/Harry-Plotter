import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import pandas as pd
from numbers_parser import Document
from os import path as path
import collections

import random


# ToDo: implement range for xlim, ylim
# ToDo: implement support for excel
# ToDo: implement regression: linear, log
# ToDo: implement log axis
# ToDO: implement barchart

class DataImport:

    def __init__(self, pathtofile, filename):
        self.pathtofile = pathtofile
        self.filename = filename

    def choose_destiny(self):
        print("not done yet :/")
        # get file extension
        # set import document according to file extension
        # set variables that are related to file; rn it is set to .numbers
        # but .column[] for example is different for Excel and numbers

    def import_data(self, sheet_nr=None, table_nr=None):
        filepath = path.join(self.pathtofile, self.filename)

        # Import for .numbers:
        doc = Document(filepath)
        sheets = doc.sheets
        tables = sheets[sheet_nr].tables
        data = tables[table_nr].rows(values_only=True)
        self.df = pd.DataFrame(data[1:], columns=data[0])


class Data:

    def __init__(self, n_datasets, row_or_column):
        self.n_datasets = n_datasets
        self.dataset_properties = {}
        self.ydata = {}  # necessary to use second dict?
        self.row_or_column = row_or_column

    def get_data(self, dataframe, first_cell, last_cell):  # better to get cells from __init__()?
        rfc, cfc = first_cell
        rlc, clc = last_cell
        self.data_table = dataframe.loc[rfc:rlc, cfc:clc]

    def make_dict(self):
        for n in range(self.n_datasets):

            if self.row_or_column == "columns":
                for col in range(1,
                                 len(self.data_table.columns)):  # starts at 1, assumes that x-data is stored in column 0
                    self.dataset_properties["data"] = self.data_table.iloc[:, col]
                    self.dataset_properties["name"] = self.data_table.columns[col]

            elif self.row_or_column == "rows":
                for row in range(1, self.data_table.shape[0]):
                    self.dataset_properties["data"] = self.data_table.iloc[row, 1:]
                    self.dataset_properties["name"] = self.data_table.iloc[
                        row, 0]  # assumes that data labels are in the first column

            else:
                raise Exception("Error: typo or wrong input, row_or_column needs to be set to column or row")

            self.ydata[n] = self.dataset_properties

    def get_xdata(self):
        if self.row_or_column == "column":
            self.xdata = self.data_table.iloc[:, 0]  # Assumption: x-data is stored in first column
        else:
            self.xdata = self.data_table.iloc[0, 1:]  # Assumption: x-data is stored in first row
            # first cell contains x-data name or can be empty

    def display_data(self):
        # make it display data after nan-removal
        # print("x-data:")
        # print(self.xdata)
        # print("y-data:")
        for data in self.ydata:
            print("x:", self.ydata[data]["xdata"])
            print(data, ":", self.ydata[data]["data"])

    def get_idx_from_values(self, ydata):
        notnan_idx = []
        iterator = iter(ydata.notna())
        for i in range(0, len(ydata)):
            if next(iterator):
                notnan_idx.append(i)
        return notnan_idx

    def get_values(self, notnanidx, ydata):
        x_corresponding = []
        y_corresponding = []
        for i in notnanidx:
            x_corresponding.append(self.xdata.iloc[i])
            y_corresponding.append(ydata.iloc[i])
        return x_corresponding, y_corresponding

    # checks ydata for nan and assigns a xdata-set to each ydata
    # if nan found, deletes it and the corresponding xdata
    def delete_nans(self):
        for dataset, properties in self.ydata.items():
            specific_dataset = self.ydata[dataset]
            ydata = specific_dataset["data"]
            if ydata.isna().any():
                notnanidxlist = self.get_idx_from_values(ydata)
                x_vals, y_vals = self.get_values(notnanidxlist, ydata)
                properties["xdata"], properties["data"] = x_vals, y_vals
            else:
                properties["xdata"] = self.xdata


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

    def cast_magic(self):
        print("Accio plot!")
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
