"""
Created on Fri Sep  6 10:02:04 2024
@author: Chenfeng Chen
"""


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

data = pd.read_csv("data.csv")


def time_to_pace(t):
    return pd.to_timedelta(t).dt.total_seconds() / data['distance'] * 100


# Compute running pace
selected_column = "male"
running_pace = time_to_pace(data[selected_column])
distance_log = np.log2(data["distance"])
track = data[data["place"] == "track"]
road = data[data["place"] == "road"]


fig = go.Figure(data=go.Scatter(
    x=data["distance"],
    y=running_pace,
    mode='markers',
    marker=dict(size=8, color='blue')
))

# Add titles and labels
fig.update_layout(
    title="Running Pace vs. Distance (Log Scale)",
    xaxis_title="Distance (meters)",
    yaxis_title="Running Pace (s/100m)",
    xaxis_type="log"
)


pio.write_html(fig, 'plot.html', auto_open=True)
