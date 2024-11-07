import streamlit as st
from sklearn.datasets import load_iris

data = load_iris(as_frame = True)
df = data.frame

st.title("Example for Streamlit")
st.header("Hello Class at UMG! ")
st.subheader("This is a subheader")

st.dataframe(df)