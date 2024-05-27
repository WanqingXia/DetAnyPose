"""
vis_rate.py

Author: Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script visualizes the success rate of the Detic algorithm.
It reads a CSV file (BOP result format) containing the results of object identification,
calculates the success rate for each object, and visualizes the results in a stacked bar chart.
"""
import pandas as pd
import matplotlib.pyplot as plt
from utils.convert import Convert_YCB


def calculate_counts(df):
    counts = df.groupby('obj_id')['score'].value_counts().unstack(fill_value=0)
    counts['total'] = counts.sum(axis=1)
    counts['success_rate'] = counts[1] / counts['total'] * 100
    return counts


def plot_results(counts, converted_obj_ids):
    # Plot the total count of each obj_id with stacked bars
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the stacked bar chart
    bars = counts[[0, 1]].plot(kind='bar', stacked=True, color=['red', 'green'], alpha=0.7, ax=ax)

    # Set x-axis labels to the converted obj_id values
    ax.set_xticklabels(converted_obj_ids, rotation=90)

    for i, obj_id in enumerate(counts.index):
        total = counts.at[obj_id, 'total']
        success_rate = counts.at[obj_id, 'success_rate']
        ax.text(i, total, f'{success_rate:.1f}%', ha='center', va='bottom')

    plt.title('Total Count and Success Rate of Each obj_id')
    plt.xlabel('object name')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.legend(['Fail', 'Success'])
    plt.tight_layout()
    plt.show()

# Main code
convert_ycb = Convert_YCB()
# Load the CSV file
file_path = './outputs/DeticSamDino_ycbv-test.csv'
df = pd.read_csv(file_path, delimiter=',')
# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()
counts = calculate_counts(df)
converted_obj_ids = [convert_ycb.convert_number(obj_id) for obj_id in counts.index]
plot_results(counts, converted_obj_ids)

