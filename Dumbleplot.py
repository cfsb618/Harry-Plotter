from sklearn.linear_model import LinearRegression

from matplotlib import ticker

import pandas as pd
from numbers_parser import Document
from os import path as path

import random


# TODO: implement support for excel
# TODO: implement regression: exponential
# TODO: implement barchart
# TODO: no nan-deletion for manual data rn
# TODO: implement column wise organised data analysis


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

    def __init__(self):
        self.ydata = []
        self.xdata = []
        self.dataset = {}
        self.properties = {}
        self.data_names = []
        self.plot_properties = []
        self.errorbars = []

    def get_data(self, column_idx, y_column_names, x_column, df):
        for i in y_column_names:
            self.ydata.append(df[i][0:column_idx+1])
            self.xdata.append(df[x_column][0:column_idx+1])
            self.data_names.append(i)

    def get_errors(self, column_idx, errorbars, y_column_name, df):
        for i in errorbars:
            if i != "None":
                self.errorbars.append(df[i][0:column_idx+1])
        
            else:
                self.errorbars.append(pd.Series(data=0))

    def construct_dict(self):
        for i in range(len(self.ydata)):
            properties = {"ydata": self.ydata[i], "xdata": self.xdata[i], "name": self.data_names[i],
                          "marker": self.plot_properties[i]["marker"], "color": self.plot_properties[i]["color"],
                          "linestyle": self.plot_properties[i]["linestyle"],
                          "regression": self.plot_properties[i]["regression"]}

            if len(self.errorbars) > 1:
                print(i)
                if self.errorbars[i].any() == 0:
                    properties["errorbars"] = None
                else:
                    properties["errorbars"] = self.errorbars[i]
            else:
                properties["errorbars"] = None

            self.dataset[i] = properties.copy()


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
