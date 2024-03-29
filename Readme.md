Hello fellow wizards

this is a simple script for (quickly) making better plots than excel or numbers. You can do the same
adjustments here as in excel, hence change marker, linestyle, axis-name and so forth.
The figure is subsequently saved to the same filepath where it took the data from, and in the fileformats specified
by the user.

### Data input
It does not matter if your data is organised in rows or columns, you just need to specify this during the script.
It also does not matter if your data contains any nans, though strings could provide a problem.
Your asked to provide the first cell and the last cell of your data to generate a data-table:
| First cell | [...] | [...] | [...] |
| --- | --- | --- | --- |
| [...] | [...] | [...] | [...] |
| [...] | [...] | [...] | last cell|

It should contain all x-values, y-values and data-labels.

#### Expectations from the script:
Data organised in rows:
![](/readme_imgs/data_in_rows.png)

Data organised in columns
![](/readme_imgs/data_in_columns.png)

At least for .numbers files, it is good practice to name every column so they can be selected easily.

### Operate script:
Best is to use script with jupyter lab.
Cells in the notebook, where the wizard needs to take action, are assigned with "Cast:"
all other cells can just be activated without the need for action.
Hint: it is a good idea to use Shift+Enter to jump through the script

To adjust all the variables like marker and linestyle for each dataset, copy the settings
with the indicated button for each dataset, then adjust accordingly.

![](readme_imgs/specify_ydata_props.png)