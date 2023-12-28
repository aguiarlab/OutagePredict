# Autoregressive Modeling of Utility Customer Outages with Deep Neural Networks

We tackle the problem of customer outage forecasting by integrating distributed temporal and spatial weather data with deep-learning prediction models. Using weather and outage data from ten random counties across New York State, we fit separate spatiotemporal models based on long short-term memory (LSTM) and convolutional neural networks (CNN) to predict customer outages over a 48-hour forecast horizon. Specifically, we consider both autoregressive and covariate-dependent signatures of variation in the development of three model architectures that predict (a) county-level outages given county-level data, (b) county-level outages given state-level data, and (c) state-level outages given state-level data. We compare our methods against statistical approaches (ARIMA, ARIMAX, and VARMAX) and a persistence-based method. The results demonstrate that our method achieves better performance over the baselines, thus providing an effective tool for electric utility companies to prepare for adverse weather events.

# Data Description
Customer outage data consists of a time series of customers without power reported every 15 minutes by utility companies across NYS counties. It was collected from 2015 to 2018. The weather data is an hourly multivariate time series of temperature, precipitation, gust, wind, and wind direction from NYS Mesonet stations. The data spans from 2016 to 2017. The customer outage data, weather data, and some leaf area index (LAI) data are merged and resampled hourly to form an hourly county-level multivariate time series within a period of two years (2016–2017), with customer outages as the target variable and weather time series as predictor variables.

We provide some code and a small sample data (Nassau County) for demonstrating the implementation of some of the time series methods used in our work. The code is built in the Jupyter Notebook environment provided by Google and can be accessed directly in Google Colab. See the following [website](https://colab.research.google.com/) for instructions on using Google Colab.


# Time Series Methods
(a) county-level outages given county-level data:
  PERSISTENCE,
  ARIMA,
  ARIMAX,
  CNN-LSTM-PC;
(b) county-level outages given state-level data:
  CNN-LSTM-SO;
(c) state-level outages given state-level data:
  VARMAX,
  CNN-LSTM-VO
