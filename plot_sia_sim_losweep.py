import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from collections import defaultdict
import argparse
import sys
import collections
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cbook import boxplot_stats
import matplotlib as mpl
from datetime import datetime
from scipy import stats
import numpy as np
import csv
from matplotlib.ticker import FormatStrFormatter
import ast
from scipy.stats import gmean

# Define a function to normalize 'avg_jct' based on 'Packed-Sticky' placement
def get_packed_sticky_avg_jct(df):
    packed_sticky_data = {}
    packed_sticky_df = df[df['placement'] == 'Packed-Sticky']
    for index, row in packed_sticky_df.iterrows():
        packed_sticky_data[row['workload']] = row['avg_jct']
    return packed_sticky_data

def _get_avg_jct(time_dict):
    """
    Fetch the avg jct from the dict
    """
    values = list(time_dict.values())
    count = 0
    jct_time = 0
    for v in values:
        jct_time += v[1] - v[0]
        count += 1

    return jct_time / count


def _get_jct_dist(time_dict):
    """
    Fetch JCT distribution
    """
    count = 0
    jct_dict = {}
    for k, v in time_dict.items():
        jct_dict[k] = v[1] - v[0]

    return jct_dict


plt.style.use('tableau-colorblind10')

# Update config according to experiment - global
temp_folder_name = ""
exp_prefix = "None"
placement = ["Default-Packed-S", "Default-Packed-NS", "Default-Random-NS", "Default-Random-S", "PMFirst", "PAL"]
placement_labels = {
    "Default-Packed-S": "Packed-Sticky",
    "Default-Packed-NS": "Packed-Non-Sticky", 
    "Default-Random-S": "Random-Sticky", 
    "Default-Random-NS": "Random-Non-Sticky",
    "PMFirst":"PMFirst",
    "PAL": "PAL"
}
# placement = ["PMFirst"]
job_ids_to_track = [0, 159]
#config = {"Fifo": [1,2,3,4,5,6,7,8]} 
config = {"Fifo": [1.0, 1.5, 2.0 , 2.5, 3.0]} 

job_data = {
    'scheduler': [],
    'placement_policy': [],
    'locality_penalty': [],
    'job_id': [],
    'jct': [],
}

for scheduler in config.keys():
    for placement_policy in placement:
        for load in config[scheduler]:
            jct_distribution = {}
            str_lf = str(load)
            stat_fname = f"{exp_prefix}_{job_ids_to_track[0]}_{job_ids_to_track[1]}_{scheduler}_AcceptAll_{placement_policy}_lf{str_lf}_job_stats.json"
            print(stat_fname)
            if os.path.exists(stat_fname):
                with open(stat_fname, "r") as fin:
                    data_job = json.load(fin)
                jct_distribution = _get_jct_dist(data_job)           

            for job_id, jct in jct_distribution.items():
                job_data['scheduler'].append(scheduler)
                job_data['placement_policy'].append(placement_policy)
                job_data['locality_penalty'].append(load)
                job_data['job_id'].append(job_id)
                job_data['jct'].append(jct)

main_df = pd.DataFrame(job_data)

print(main_df)


placement_policies = ["Default-Packed-S", "Default-Packed-NS", "Default-Random-NS", "Default-Random-S", "PMFirst", "PAL"]

plot_data = {
    "placement" : [],
    "locality_penalty": [],
    "avg_jct": [],
    "p99_jct": [],
}

workload = [1.0, 1.5, 2.0, 2.5, 3.0]
placement_jct = {}
placement_wait_time = {}
for i, placement in enumerate(placement_policies):
    avg_jcts = []
    avg_wait_times = []
    data = main_df[main_df['placement_policy'] == placement]
    for load in config["Fifo"]:
        data_load = data[data['locality_penalty'] == load]
        avg_jcts.append(data_load['jct'].mean())
        plot_data['avg_jct'].append(data_load['jct'].mean())
        plot_data['p99_jct'].append(data_load['jct'].quantile(0.99))
        plot_data['locality_penalty'].append(load)
        plot_data['placement'].append(placement_labels[placement])
    placement_jct[placement] = avg_jcts

df_jct = pd.DataFrame(plot_data)


print(df_jct)
# Renaming 'PMFirst' to 'PM-First' in 'placement' column
df_jct['placement'] = df_jct['placement'].replace('PMFirst', 'PM-First')
df_jct['avg_jct'] = df_jct['avg_jct'] / 3600 # convert to hrs

# Graphing customizability options
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('seaborn-whitegrid')
sns.set_palette("colorblind") 

placement_labels = ["Packed-Sticky", "Packed-Non-Sticky",  "Random-Sticky", "Random-Non-Sticky", "PM-First", "PAL"]


# Sort the DataFrame for better presentation
df_jct.sort_values(by=['placement', 'locality_penalty'], inplace=True)

# print(result_df)
# result_df.to_csv("geomean.csv")

#Define RGB hex values for colors - from tableau colorblind palette
colors = {
    "Packed-Sticky": "#1170aa",
    "Packed-Non-Sticky": "#5fa2ce",
    "Random-Sticky": "#7b848f",
    "Random-Non-Sticky": "#a3acb9",
    "PM-First": "#fc7d0b",
    "PAL": "#c85200"
}

# Define custom order for placement policies
custom_order = ["Random-Non-Sticky", "Random-Sticky","Packed-Non-Sticky", "Packed-Sticky", "PM-First", "PAL"]

plt.figure(figsize=(10,4))
# Sort the dataframe based on avg_jct in descending order
g = sns.catplot(
    x="locality_penalty",
    y="avg_jct",
    hue="placement",
    hue_order=custom_order,
    legend=False,
    palette=colors,
    data=df_jct,
    kind="bar",
    height=2.8,  # Adjust the height of each facet
    aspect=4,  # Adjust the width of each facet
)

plt.legend(bbox_to_anchor=(0.5, 1.3),loc='upper center', ncol=3, fontsize=17)

#plt.tight_layout()
plt.xlabel('Locality Penalty')
plt.ylabel('Avg JCT (hours)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(f"sia-philly-avg-jct-losweep.pdf",
            format="pdf", dpi=600, bbox_inches="tight")