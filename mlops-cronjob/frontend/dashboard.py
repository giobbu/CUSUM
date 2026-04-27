import pathlib
import plotly.graph_objects as go
import streamlit as st
import pickle as pkl
from loguru import logger
import glob

dir_path = pathlib.Path(__file__).parent.parent
print(dir_path)

#results_path = dir_path / "data" / "2026-03-08" /"detection_results_13-03-25.pkl"

results_files = glob.glob(str(dir_path / "data" / "*" / "detection_results.pkl"))

# get the latest file based on modification time
if results_files:
    latest_file = max(results_files, key=lambda x: pathlib.Path(x).stat().st_mtime)
    results_path = latest_file
    logger.info(f"Latest results file found: {results_path}")

with open(results_path, "rb") as f:
    results = pkl.load(f)

observations = results.get("observations", [])
change_points = results.get("change_points", [])
positive_changes = results.get("pos_changes", [])
negative_changes = results.get("neg_changes", [])
metadata = results.get("metadata", {})


# add title for streamlit app
st.title("CUSUM Change Point Detection Results")

# Create a line plot of observations with change points marked
# large size for better visibility
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[i for i in range(len(observations))],
        y=observations,
        mode="lines",
        name="Observations"
    )
)
# add change points as vertical lines
for cp in change_points:
    fig.add_shape(
        type="line",
        x0=cp,
        y0=min(observations),
        x1=cp,
        y1=max(observations),
        line=dict(color="red", width=2, dash="dash"),
    )
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# add figure with positive and negative changes
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=[ i for i in range(len(positive_changes))],
        y=[ pos[0] for pos in positive_changes],
        name="Positive Changes",
        mode="lines",
        line=dict(color="green", width=2)


    )
)
fig2.add_trace(
    go.Scatter(
        x=[ i for i in range(len(negative_changes))],
        y=[ neg[0] for neg in negative_changes],
        name="Negative Changes",
        mode="lines",
        line=dict(color="orange", width=2)
        
    )
)
#add horizontal line for threshold
threshold = metadata.get("threshold", 0)
fig2.add_shape(
    type="line",
    x0=0,
    y0=threshold,
    x1=len(positive_changes),
    y1=threshold,
    line=dict(color="red", width=2, dash="dash"),
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
