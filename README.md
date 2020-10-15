# Investment-Aid-using-Deep-Learning-and-NLP
Developing a Stock investment portfolio system that predicts stock prices while accounting for company news data

 - Collected and performed data engineering on the dynamic NIFTY-50 Stock Market Dataset that had Open, High, Low, Close, and Turnover values of 50 most influential Indian companies from June 2000 to June 2020 and ongoing. 
 - Trained a baseline linear regression model that obtained an accuracy of 65 %. 
 - LSTMs neural networks are special types of Recurrent neural networks that have loops in them, allowing information to persist. This means that not only were they able to update the weights of their nodes frequently, but they were also able to take into consideration previous weights, which makes them ideal for time-series data such as stock price data.
 - With a 9-layer neural network, having 4 50-node LSTM layers followed alternatively by 4 0.2 dropout layers and the last layer being a Dense layer with adam optimizer and mean squared error as the loss function, the model trained over 100 epochs to give an RMSE value of 13.896. 
 - To incorporate an NLP model to assess twitter or news data we have so far tried to generate a time series dataset of all the news or twitter feed related to each of the companies in the NIFTY-50 dataset and append the news entry of that day with the respective stock price of that company in the dataset. 
 - On this new dataset, we used sentiment analysis models like Bag-of-Words as a baseline and chose to test with Word2Vec, BERT, NLTIK, and ERNIE to generate scores on a scale of +10 to -10 with +10 being extremely positive for the stock’s health to -10 being the worst. 
 - The entire system updates its news scores when stock prices correlated to a particular type of news reaches new extremes. 
 - Unique because it is designed for the Indian stock market and news and deployed on a cross-platform framework for the novice investor, democratizing stock market investment and reducing reliance on other players. 
 
 ### Challenges:
 - Preparing a dataset of the Indian stock market on the Nifty-50 index, 
 - Correlate with the corresponding news/twitter feed of that company for that day,
 - Building a language processing model that dynamically generates vector scores of the news’/twitter feeds’ effect on the stock price and updates the LSTM model accordingly 

### Further Work
We intend to build a pipeline for this entire process to work in flow with real-time daily stock price data and news data as input which would pass from the feature engineering and correlation module to the feature loading model, and finally to the model training and testing modules. The models will be trained and updated on a GPU accelerated platform, like Kaggle’s or on Google Colab.

