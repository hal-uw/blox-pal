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


# Graphing customizability options
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('seaborn-whitegrid')
sns.set_palette("colorblind")  

placement_labels = {
    "Default-Packed-S": "Packed-Sticky",
    "Default-Packed-NS": "Packed-Non-Sticky", 
    "Default-Random-NS": "Random-Non-Sticky", 
    "Default-Random-S": "Random-Sticky", 
    "PMFirst":"PMFirst",
    "PAL": "PAL"
}

colors_dict = {
   "Default-Packed-S": "#1170aa", 
   "Default-Packed-NS": "#5fa2ce", 
   "Default-Random-NS": "#a3acb9", 
   "Default-Random-S": "#7b848f", 
   "PMFirst": "#fc7d0b", 
   "PAL":"#c85200"
}

scheduler="Fifo"

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
job_ids_to_track = [0, 10]
#config = {"Fifo": [1,2,3,4,5,6,7,8]} 
config = {"Fifo": [8.0, 10.0, 12.0, 14.0]} 

job_data = {
    'scheduler': [],
    'placement_policy': [],
    'job_load': [],
    'job_id': [],
    'jct': [],
}

for scheduler in config.keys():
    for placement_policy in placement:
        for load in config[scheduler]:
            jct_distribution = {}
            stat_fname = f"{exp_prefix}_{job_ids_to_track[0]}_{job_ids_to_track[1]}_{scheduler}_AcceptAll_{placement_policy}_load_{load}_job_stats.json"
            print(stat_fname)
            if os.path.exists(stat_fname):
                with open(stat_fname, "r") as fin:
                    data_job = json.load(fin)
                jct_distribution = _get_jct_dist(data_job)           

            for job_id, jct in jct_distribution.items():
                job_data['scheduler'].append(scheduler)
                job_data['placement_policy'].append(placement_policy)
                job_data['job_load'].append(load)
                job_data['job_id'].append(job_id)
                job_data['jct'].append(jct)

main_df = pd.DataFrame(job_data)

print(main_df)

# Aggregate main_df to return avg JCT values
data_fifo = {
    "placement_policy": [],
    "job_load": [],
    "avg_jct_hrs": []
}


# filter out this scheduling policy
filtered_main = main_df[main_df["scheduler"] == scheduler]
placement_policies = filtered_main["placement_policy"].unique()
placement_jct = {}
for i, plac in enumerate(placement_policies):
    avg_jcts = []
    data = filtered_main[filtered_main["placement_policy"] == plac]
    for load in config[scheduler]:
        data_load = data[data["job_load"] == load]
        avg_jcts.append(data_load["jct"].mean()/3600.0) #to convert to hours
        data_fifo['placement_policy'].append(plac)
        data_fifo['job_load'].append(load)
        data_fifo['avg_jct_hrs'].append(data_load["jct"].mean()/3600.0)
    placement_jct[plac] = avg_jcts

# Sort placements based on their average JCT values in descending order
sorted_placements = sorted(placement_jct.items(), key=lambda x: np.mean(x[1]), reverse=True)



df_fifo = pd.DataFrame(data_fifo)
print(df_fifo)

# Drop rows with NaN values in avg_jct_hrs
df_fifo = df_fifo.dropna(subset=['avg_jct_hrs'])

# Plot using seaborn
sns.lineplot(data=df_fifo, x='job_load', y='avg_jct_hrs', hue='placement_policy', marker='o',linestyle="--",palette=colors_dict)

plt.legend(loc='upper left')
plt.legend(title='Placement Policy', labels=[placement_labels[policy] for policy in df_fifo['placement_policy'].unique()], loc='upper left')

plt.tight_layout()
plt.xlabel("Job Load (jobs/hour)")
plt.ylabel("Avg JCT (Job Completion Time) in hours")
# plt.title(
#     "Job Load Scaling with Synergy trace, FIFO Scheduler"
# )
plt.savefig(f"synergy-{scheduler}-jct-vs-job-load.pdf",
            format="pdf", dpi=600, bbox_inches="tight")




