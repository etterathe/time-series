import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py


st.title('Time Series Forecast App')

st.markdown("""
This app performs time series forecast using prophet algorithm.
""")


upload = st.file_uploader("Upload a file", type="csv")


def app():
    if upload:
        data = pd.read_csv(upload)
        st.sidebar.header("Select the number of days to forecast")
        numbers = st.sidebar.selectbox('No. of days', list(range(10, 366)))
        model = Prophet()
        model.fit(data)
        future_df = model.make_future_dataframe(periods=numbers)
        forecast = model.predict(future_df)
        plot = model.plot(forecast)
        st.write(plot)


if __name__ == '__main__':
    app()