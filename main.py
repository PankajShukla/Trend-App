
import pandas as pd
import os

import numpy as np
import seaborn as sns
import datetime
sns.set_style('whitegrid')
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf
from tensorflow import keras


from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras import callbacks

import h5py

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_column", 500)


filename = 'Input_data.csv'
df = pd.read_csv(filename)
df.head(3)


df['timestamp'] = pd.to_datetime(df['Date-Time'], format='%d/%m/%y %H:%M')

df['date'] = df['timestamp'].dt.date

df['dayofweek'] = df['timestamp'].dt.weekday
df['dayofmonth'] = df['timestamp'].dt.day
df['year'] = df['timestamp'].dt.year
df['quarter'] = df['timestamp'].dt.quarter
df['month'] = df['timestamp'].dt.month
df['weekofyear'] = df['timestamp'].dt.isocalendar().week
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

df['prev_Close'] = df['Close'].shift(1)
df['prev_Close'].fillna(df['Close'], inplace=True)

df['TR1'] = df['High']-df['Low']
df['TR2'] = df['High']-df['prev_Close']
df['TR3'] = df['Low'] -df['prev_Close']

df['TR'] = df[['TR1','TR2','TR3']].max(axis=1)

df['ATR'] = df['Close'].rolling(14, min_periods=1).mean()
df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
df['EMA_50'] = df['Close'].ewm(com=2).mean()


df.drop(columns=['prev_Close', 'TR1', 'TR2', 'TR3'], inplace=True)
df = df[~((df.Trend_initial==1) & (df.Trend_final==0))]
df_2024 = df[df.year==2024]

output_var = pd.DataFrame(df_2024['Trend_final'])
features = ['Close', 'cci_50', 'Close_supertrend_diff', 'ATR_EMA_diff', 'TR']

def variable_scaling( _df, features, output_var):

  #Scaling
  scaler = MinMaxScaler()
  feature_transform = scaler.fit_transform(_df[features])
  feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=_df.index)
  feature_transform.head()

  return feature_transform


def generate_classification_report(_model, X_test, y_test ):
  testPredict = _model.predict(X_test)
  yPred = testPredict.ravel()

  yPred_ = []
  for v in yPred:
    if v>=0.5:
      yPred_.append(1)
    else:
      yPred_.append(0)

  print(classification_report(yPred_, y_test))
  return yPred_


def set_up_streamlit():
    # -------------------------------------------------
    # Set up for streamlit
    # -------------------------------------------------

    st.set_page_config(
        page_title="Home",
        page_icon="👋",
        layout="wide"
    )
    st.markdown(f'<p style=font-size:50px;border-radius:2%;"> Trend Prediction Application</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        selected_dt = st.date_input("Select a date to view Trend", datetime.date.today() - datetime.timedelta(20))
    with c2:
        pass

    return selected_dt



def main(_df):

    """
    ------------------------------------------------------------
    Load Stored Model
    ------------------------------------------------------------
    """
    LSTM_model = load_model('stock_trend_lstm_2024-06-10_H17.h5')

    print(_df.shape)
    X_try_feature_transform = variable_scaling(_df, features, output_var)

    X_try = np.array(X_try_feature_transform).reshape(X_try_feature_transform.shape[0], 1,
                                                      X_try_feature_transform.shape[1])
    y_try = pd.DataFrame(_df['Trend_final']).values.ravel()
    print(X_try.shape, y_try.shape)

    scores = LSTM_model.evaluate(X_try, y_try, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    y_try_pred = generate_classification_report(LSTM_model, X_try, y_try)
    _df = pd.merge(_df, pd.DataFrame(y_try_pred, columns=['predicted_TrendFinal']), on=_df.index, how='left')
    _df['Close_TrendFinal'] = np.where(_df['Trend_final'] == 1, _df['Close'] * (1 - 0.0003), np.nan)
    _df['Close_Precicted_TrendFinal'] = np.where(_df['predicted_TrendFinal'] == 1, _df['Close'] * (1 + 0.0003), np.nan)

    return _df



if __name__ == "__main__":
    # Execute the main function if the script is run directly

    selected_dt = set_up_streamlit()

    output_data = main(df_2024)

    data = output_data[output_data.date == selected_dt]

    fig, axes = plt.subplots(1, 1, figsize=(14, 5))
    sns.lineplot(data=data, x='timestamp', y='Close', ax=axes)
    sns.lineplot(data=data, x='timestamp', y='EMA_50', ax=axes, color='lightgreen')
    sns.scatterplot(data=data, x='timestamp', y='Close_TrendFinal', ax=axes, color='red')
    sns.scatterplot(data=data, x='timestamp', y='Close_Precicted_TrendFinal', ax=axes, color='purple')

    st.pyplot(fig)




