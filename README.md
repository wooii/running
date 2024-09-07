## Running Pace Analysis and Prediction
1. Pace is linearly related to log₂(Distance).
2. Two distinct phases: 200m to 1500m and 1500m to 42195m (marathon).
3. Each phase follows its own linear formula: Pace = a * log₂(Distance) + b.
4. Use this model to predict or evaluate your personal best for various distances.

To test the app locally:
1. Open a terminal, navigate to the folder containing app.py using cd.
2. Run the app with the command streamlit run app.py.