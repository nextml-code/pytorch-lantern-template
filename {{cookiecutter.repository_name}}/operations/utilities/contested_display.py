import operations
import data
import pandas as pd

import streamlit as st

st.set_page_config(layout="wide")


def reset_index():
    st.session_state.index = 0


def next():
    st.session_state.index = min(st.session_state.index + 1, len(dataframe))


def previous():
    st.session_state.index = max(st.session_state.index - 1, 0)


datasets = data.datasets()
option = st.selectbox("Select dataset", datasets.keys(), on_change=reset_index)
dataframe = pd.read_csv(f"contested/{option}.csv")


if "index" not in st.session_state:
    reset_index()

st.subheader(f"index: {st.session_state.index}")
col1, col2, _, _ = st.columns([0.1, 0.17, 0.1, 0.63])
col1.button("<- previous", on_click=previous)
col2.button("next ->", on_click=next)
row = dataframe.iloc[st.session_state.index]
image = datasets[option][row["dataset_index"]].representation()

st.image(image, width=1200)
st.markdown(
    dataframe[st.session_state.index : st.session_state.index + 1].to_html(
        escape=False
    ),
    unsafe_allow_html=True,
)
