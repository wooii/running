import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import datetime
import plotly.express as px


class RunningPaceApp:
    def __init__(self, default_csv="default_data.csv"):
        self.default_csv = default_csv
        if "data" not in st.session_state:
            self.reset_data()
        self.data = st.session_state["data"]
        self.scale_option = st.sidebar.radio("Choose x-axis scale", ["Log", "Linear"])
        self.pace_type = st.sidebar.radio("Choose y-axis pace type", ["s/100m", "mm:ss/km"])
        self.show_trendline = st.sidebar.checkbox("Show Trend Lines", True)
        self.selected_columns = st.sidebar.multiselect(
            label="Select columns to plot",
            options=self.columns_to_plot,
            default=self.columns_to_plot[:3])
        self.selected_distances = st.sidebar.multiselect(
            label="Select Distances",
            options=self.unique_distances,
            default=self.unique_distances[:-2])
        self.handle_pb_input()

    @property
    def columns_to_plot(self):
        return self.data.columns.tolist()[3:]

    @property
    def unique_distances(self):
        return sorted(self.data["distance"].unique())

    def reset_data(self):
        st.session_state["data"] = pd.read_csv(self.default_csv)
        st.success("Data reset to default.")

    def save_data(self):
        st.session_state["data"].to_csv("data.csv", index=False)
        st.success("Data saved as 'data.csv'")

    def handle_pb_input(self):
        if "my_pb" in self.selected_columns:
            for dist in self.selected_distances:
                time_input = st.sidebar.text_input(
                    f"Your time for {dist}m (hh:mm:ss)",
                    self.data.loc[self.data["distance"] == dist, "my_pb"].values[0])
                self.data.loc[self.data["distance"] == dist, "my_pb"] = time_input

    def finish_time_to_pace(self, t):
        t = t.dropna()
        seconds_per_meter = pd.to_timedelta(t).dt.total_seconds() / self.data["distance"]
        return seconds_per_meter * (100 if self.pace_type == "s/100m" else 1000)

    def format_pace(self, seconds):
        if pd.isna(seconds):
            return None
        return datetime.datetime.min + datetime.timedelta(seconds=seconds)

    def add_trend_line(self, column, color, name, distance_range):
        trendline_data = self.data[(self.data["distance"] >= distance_range[0])
                                   & (self.data["distance"] <= distance_range[1])]
        if trendline_data.empty:
            return None

        pace = self.finish_time_to_pace(trendline_data[column])
        distance_log2 = np.log2(trendline_data["distance"])

        valid_indices = ~pace.isna() & ~distance_log2.isna()
        pace, distance_log2 = pace[valid_indices], distance_log2[valid_indices]

        if pace.empty or distance_log2.empty:
            return None

        model = LinearRegression(positive=True).fit(distance_log2.values.reshape(-1, 1), pace)
        x_range = np.linspace(distance_range[0], distance_range[1], 100)
        y_pred = model.predict(np.log2(x_range).reshape(-1, 1))

        if self.pace_type == "mm:ss/km":
            y_pred = [self.format_pace(p) for p in y_pred]

        r2 = r2_score(pace, model.predict(distance_log2.values.reshape(-1, 1)))
        formula = (
            f"P = {model.coef_[0]:.2f}log₂D {'-' if model.intercept_ < 0 else '+'} "
            f"{abs(model.intercept_):.2f}, R²={r2:.3f}"
        )

        # Create hover text with both formula and x, y data points
        hover_text = [
            f"Distance: {dist:.0f}m<br>Pace: {p.strftime('%M:%S')} /km<br>{formula}"
            if self.pace_type == "mm:ss/km" else
            f"Distance: {dist:.0f}m<br>Pace: {p:.2f} s/100m<br>{formula}"
            for dist, p in zip(x_range, y_pred)
        ]


        return go.Scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            line=dict(color=color, width=2),
            name=name,
            text=hover_text,
            hoverinfo="text",
            showlegend=False,
            opacity=0.8
        )

    def plot_data(self):
        if self.selected_distances:
            st.write("""
            ## Running Pace Analysis and Prediction
            1. Pace is linearly related to log₂(Distance).
            2. Two distinct phases: 200m to 1500m and 1500m to 42195m (marathon).
            3. Each phase follows its own linear formula: Pace = a * log₂(Distance) + b.
            4. Use this model to predict or evaluate your personal best for various distances.
            """)

            data_filtered = self.data[self.data["distance"].isin(self.selected_distances)]
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly

            for idx, column in enumerate(self.selected_columns):
                color = colors[idx % len(colors)]  # Use modulo to cycle through colors
                pace = self.finish_time_to_pace(data_filtered[column])
                if self.pace_type == "mm:ss/km":
                    pace = [self.format_pace(p) for p in pace]
                else:
                    pace = pace.tolist()

                # Plot the data points
                fig.add_trace(go.Scatter(
                    x=data_filtered["distance"],
                    y=pace,
                    mode="markers",
                    marker=dict(size=6, color=color),
                    name=f"{column}"
                ))

                if self.show_trendline:
                    for distance_range in [(200, 1500), (1500, 42195)]:
                        trend_line = self.add_trend_line(column, color, column, distance_range)
                        if trend_line:
                            fig.add_trace(trend_line)

            fig.update_xaxes(type="log" if self.scale_option == "Log" else "linear",
                             title="Distance (meters)")

            if self.pace_type == "mm:ss/km":
                fig.update_yaxes(
                    title="Pace (mm:ss/km)",
                    tickformat="%M:%S",
                    rangemode="tozero"
                )
            else:
                fig.update_yaxes(
                    title="Pace (s/100m)",
                    rangemode="tozero"
                )

            fig.update_layout(
                title={'text': "Running Pace vs Distance"},
                legend=dict(
                    yanchor="top",
                    y=1.05,
                    xanchor="left",
                    x=0.05,
                    orientation="v"
                ),
                height=600
            )

            st.plotly_chart(fig)
        else:
            st.write("Please select at least one distance.")

    def upload_data(self):
        uploaded_file = st.file_uploader("Upload a new CSV", type="csv")
        if uploaded_file:
            st.session_state["data"] = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully.")

    def display_buttons(self):
        st.button("Reset to Default", on_click=self.reset_data)
        st.button("Save Current Data", on_click=self.save_data)

    def display_data(self):
        st.write(self.data)

    def run(self):
        self.plot_data()
        self.upload_data()
        self.display_buttons()
        self.display_data()


# %% Run the app
self = RunningPaceApp()
self.run()
