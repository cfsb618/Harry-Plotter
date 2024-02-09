import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from numbers_parser import Document
from os import path as path
import collections


class DataImport:

    def handle_data(self):
        print("not done yet :/")
        # get file extension
        # set import document according to file extension
        # set variables that are related to file; rn it is set to .numbers
        # but .column[] for example is different for Excel and numbers


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
        print(self.data_table)

    def make_dict(self):
        for n in range(self.n_datasets):

            if self.row_or_column == "column":
                for col in range(1, len(self.data_table.columns)):  # starts at 1, assumes that x-data is stored in column 0
                    self.dataset_properties["data"] = self.data_table.iloc[:, col]
                    self.dataset_properties["name"] = self.data_table.columns[col]

            elif self.row_or_column == "row":
                for row in range(1, self.data_table.shape[0]):
                    self.dataset_properties["data"] = self.data_table.iloc[row, 1]
                    self.dataset_properties["name"] = self.data_table.iloc[
                        row, 0]  # assumes that data labels are in the first column

            else:
                raise Exception("Error: typo or wrong input, row_or_column needs to be set to column or row")

            self.ydata[f"y{n}"] = self.dataset_properties

    def get_xdata(self):
        if self.row_or_column == "column":
            self.xdata = self.data_table.iloc[:, 0]  # Assumption: x-data is stored in first column
        else:
            self.xdata = self.data_table.iloc[0, 1:]  # Assumption: x-data is stored in first row
            # first cell contains x-data name or can be empty

    def display_data(self):
        print("x-data", self.xdata)
        print("y-data:")
        for data in self.ydata:
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
        for data, properties in self.ydata.items():
            ydata = properties["data"]
            if ydata.isna().values().any():
                notnanidxlist = self.get_idx_from_values(ydata)
                x_vals, y_vals = self.get_values(notnanidxlist, ydata)
                properties["xdata"], properties["data"] = x_vals, y_vals
            else:
                properties["xdata"] = self.xdata
                #





