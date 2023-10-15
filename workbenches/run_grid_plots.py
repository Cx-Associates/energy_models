'''
Goal is to make nice visualization showing carbon intensity of the grid over time.
'''

import os
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})

## for interactive plotting while debugging in PyCharm
plt.interactive(True)
mpl.use('TkAgg')

# set parent directory and filepaths
parent_dir = os.path.dirname(os.getcwd())
path_emissions_data = os.path.join(parent_dir, 'data', 'Region_NE_cleaned.csv')

# now read emissions data and then join it with usage data
df = pd.read_csv(
    path_emissions_data,
    usecols=['Local time', 'CO2 Emissions Intensity for Consumed Electricity'],
    index_col='Local time',
    parse_dates=True
)
df.rename(columns={'CO2 Emissions Intensity for Consumed Electricity': 'emissions intensity'}, inplace=True)

# deal with DST - probably better to localize both dfs but I don't have time for that rn
df = df[~df.index.duplicated(keep='first')]

# make quarter column for sns plotting
df['quarter'] = df.index.year.astype(str) + ' Q' + df.index.quarter.astype(str)
df = df[df['quarter'] != '2023 Q4']

# initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="quarter", hue="quarter", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "emissions intensity",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "emissions intensity", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "emissions intensity")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

pass