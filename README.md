# Investment Aid using Deep Learning and NLP
Developing a Stock investment portfolio system that predicts stock prices while accounting for company news data

## Table of Contents
- [Introduction](#introduction)
- [Project Description](#project-description)
- [Literature Survey](#literature-survey)
- [Planning and Development Process](#planning-and-development-process)
- [Project Timeline](#project-timeline)
- [Project Deliverables](#project-deliverables)
- [Project So Far](#project-so-far)
  - Architecture
  - Kaggle Notebooks
  - Notion Notebook(s)
- [Presentations](#presentations)
- [Challenges](#challenges)
- [Further Work](#further-work)

## Introduction
The idea of “The Stock Market” is a very mysterious, chaotic, risky, and complicated concept to an absolute layman. An individual with knowledge and insight of how the stock market fluctuates, is either trained exhaustively in this domain or has years of experience. Wikipedia defines the stock market as,
“ A stock market, equity market or share market is the aggregation of buyers and sellers of stocks (also called shares), which represent ownership claims on businesses; these may include securities listed on a public stock exchange , as well as stock that is only traded privately, such as shares of private companies
which are sold to investors through equity crowdfunding platforms. ” <br>
![enter image description here](https://riskmanagementguru.com/wp-content/uploads/2018/10/video.gif)

To a common citizen wanting to invest in the stock market, terms like “ownership claims”, “securities”, “private trading”, “equity” etc.
only makes literal sense. <br>

One definitely knows that the stock market game is meant for experienced and big players, and most of the time the new investors who take a chance with their luck, are often let down by the entire dynamic and working of the system. Investors who are new to this arrangement often cling to stock brokers, follow influential players, or even blindly trust a “hot tip”. This has been the tradition ever since stock exchanges were made available to the common man. Now in the data age, with Artificial Intelligence, Deep Learning, Data Analytics, and Machine Learning, stock market valuation and growth can be predicted based on real and relevant information. <br>

There has been an entire ocean of research based on the integration of computer based decisions and the stock market. Many people have analysed the parameters that affect the stock exchange. Based on these parameters, many models have also been designed in order to get the best fit prediction. <br>
As a team, we aim not only to add to that ocean of research but also conclude which among the existing models and approaches are most likely to yield in profitable results. <br> 

The motivation of this project stems out of wanting to bring order to the existing chaos. <br>
Which brings us to the problem statement of our project: <br>

> *To design a machine learning model using deep learning and sentiment analysis (on market news), for
predicting the future trends in the stock market.**

[Back to Top](#table-of-contents)


## Project Description
As a team we were inspired to use machine learning as the foundation of this project from the common course of Artificial Intelligence and Machine Learning through Coursera that all four of us opted for, in the beginning of our pre-final year. Having understood the basic models and working the provided datasets, there is a common excitement and eagerness to learn and dig deeper. Apart from this, another homogeneity within the group was the low level of financial knowledge in terms of the stock market. The idea of combining something that we’re well aware of along with something we’re not so well versed with( but are definitely interested in learning about), was sealed after we did some relevant reading, and realistically measured where this project would go. <br>

Unique because it is **designed for the Indian stock market and news** and **deployed on a cross-platform framework** for the novice investor, **democratizing stock market investment** and reducing reliance on other players. 
[Back to Top](#table-of-contents)


## Literature Survey

| Citation                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Inference                                                                                                                                                                                                                                                                                            |   |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| Aditya Bhardwaj et\. al, 2015 | In this paper, different sentiment classification methodologies are discussed\. There are broadly two approaches \- Lexicon Based Approach and Machine Learning Approach Majority of the classification techniques are types of Supervised Learning, which falls under the Machine Learning Approach Under Supervised Learning, there are different techniques like Probabilistic Classifier\(Naive Bayes\), Decision Tree Classifier, Rule Based Classifier and Linear Approach \(Support Vector Machine and Neural Network\)                                                                                                                                                                                                         | On implementation of several classification algorithms and upon comparison, it is observed that generally SVM \(Support Vector Machine\) yields results with better efficiency\.                                                                                                                     |   |
| Dou Wei, 2019                 | In this paper, to predict the stock price trend, an Attention\-based Long Short Term Memory model is proposed, consisting of four layers: Input Layer, Hidden Layer, Attention Layer and Output Layer\. Input Layer accepts input data like Date, Opening, Closing, Maximum and Minimum Price for a stock\. Hidden Layer is formed of LSTM which is affected by input data of the current moment and previous moment Attention layer calculates the weights of the input data and learns from it\. Output Later predicts the stock trend on N\+1 day for an input data of N days                                                                                                                                                       | For a small time frame, there is a little time delay between the predicted value and the real value\. But with the increase in the time frame, this time delay is minimised considerably\. Hence we should go for an input data of longer time frame \(preferably greater than or equal to 2 years\) |   |
| Saloni Mohan et\. al, 2019    | In this paper, We calculate accuracy of stock price predictions by gathering a large amount of time series data and analyzing it in relation to related news articles, using deep learning models\. News data is collected by using deep learning models like web scraper and sentiment is identified to check their effect on stock prices\. Mean Absolute Percentage Error\(MAPE\) is used to calculate the average relative error of the predictions, as a percentage\. They have used ARIMA models, and RNN\- LSTM models with different approaches to get respective results\. The model with input values as the previous price data and the text polarity of the financial new data performed very well with least MAPE value\. | While working on data within a small range of 5 years it is seen that when RNN\-LSTM model with an approach where the stock data of previous years and the text polar data from the sentimental analysis is the best approach with the least error in values\.                                       |   |
| Tejas Mankar, et\. al, 2018   | In this paper, the sentiment analysis from Tweets is used to predict the trend of the stock\. The data for Sentiment Analysis is fetched from Twitter using Twitter API and it is cleaned using NLTK library, by removing stop words, symbols like @ and \#\. Since a huge dataset results in a huge number of features \(different words from tweets\), N most significant features were selected to train the classifier model using Naive Bayes and Support Vector Machine\. Then the correlation between the Twitter sentiment and Stock Market prices is calculated and then the Stock Price Prediction is carried out\.                                                                                                          | Daily up and down stock price closing value changes is projected with an accuracy of 87\.6%\.                                                                                                                                                                                                        |   |
[Back to Top](#table-of-contents)
## Planning and Development Process

We do understand that strategic planning is the key to a successful project, but with the nature of our project, all our plans are subject to flexible change. Irrespective of the minute changes that we will see during the development of this system, there are going to be a few underlying steps that will remain
constant. <br>
The model of work that we will follow is derived from the **spiral model of software development.** The spiral model is one of the most flexible software development life cycle methodologies which takes a cue from the iterative model. Instead of going with the pre-defined phases, we would like to go with the four phases that we think will suit this project. 
The phases are: <br> 
  1. Research Phase <br>
  2. Design Phase <br>
  3. Execution or Implementation Phase <br>
  4. Testing Phase <br> <br>
  
  ![Project Development Process](images/planning%201.JPG)
The project passes through four phases (Research, Design, Execution and Testing) over and over in a “spiral” until completed, allowing for multiple rounds of refinement.
This model allows for the building of a **highly customized product**, and **user feedback can be incorporated from early on** in the project. Since our project has 2 major phases in terms of delivery, this is the approach we would like to follow. <br>

Adhering to the spiral model, our research will have two major wings, one being understanding the application of deep learning concepts in finance and stock market prediction. The other being an absolute focus on stock market fluctuations and predictions. With constant learning and research of both these domains, we will be able to tweak the project whenever required. <br>

Our design phase will be in sync with the research that we will be looking into. As we learn more and more about the different algorithms that can be used and the parameters that cause stocks to rise and fall, we will be able to design a system according to the output we require. We will start off by working on the design of the machine learning model and then move towards designing the user interface. <br>

The execution phase of our project will have 3 sub parts. <br>
We plan to begin with the implementation of the sentiment analysis model. The sentiment analysis model will be trained to conclude the sentiment and the impact of news about companies, industries and players, on the stock market. We will work towards making sure that the model developed is able to judge the sentiment of the data correctly and contextually. <br>

The second leg of the execution phase will be the implementation of stock market prediction using historical data. The results of the sentiment analysis done on the news dataset will be appended to the stock market dataset. This will give us a new dataset for us to train our model on. Using deep learning,
along with different libraries, APIs, GPUs/TPUs, we will aim to get the most optimal output.<br>

The last phase of the execution phase would be integrating this stock prediction model based on machine learning with the user interface. This phase would focus on user experience, as well making the system easy to use after all. <br>

The testing phase of our project will not be statically allocated in terms of the time frame. We intend to develop and test one after the other, as it yields in substantial and meaningful changes. Also making changes, adding and removing components are best done when it is done at the basic engineering level. <br>

We will be testing the sentiment analysis model, the stock prediction model as well as the user interface as and when we build the components one by one.
[Back to Top](#table-of-contents)

## Project Timeline

The project timeline is essential to ensure that the project is transparent and well-planned. At the end of the day it has a direct effect on the work we put in. A conventional timeline would see the events of the projects arranged in a chronological manner. We would like to set our timeline by allocating a number of days in order to make sure the project has a specific end date. <br>
![Project Timeline Gantt Chart](images/timeline%201.JPG)

Having understood what this project will need in terms of time, energy and effort, we have allocated a number of days as per the major tasks defined previously. <br>

| Sr\. No | Task                    | Agenda                                                                                                                                         | Period  |
|---------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| 1       | Research and Study      | Explore the existing stock price prediction and sentiment analysis system more extensively\. Study of machine learning models and stock market | 40 days |
| 2       | Implementation: Phase 1 | Design and implementation of Sentiment Analysis Model                                                                                          | 50 days |
| 3       | Dataset Integration     | Engineering dataset for Stock Price Prediction by appending results from Implementation: Phase 1                                               | 10 days |
| 4       | Implementation: Phase 2 | Design and development of Stock Price Prediction Model                                                                                         | 20 days |
| 5       | Analysis and Testing    | Testing and debugging the system with a bottom\-up approach                                                                                    | 25 days |
| 6       | Implementation: Phase 3 | Design and development of User interface 30 days 7\.                                                                                           | 30 days |
| 7       | Final Documentation     | Black book compilation                                                                                                                         | 20 days |

[Back to Top](#table-of-contents)

## Project Deliverables
As a team working on such a project, we realise that a “project deliverable” is something more definite then the casual meaning. It is a specific output created as a result of work performed during the course of the project. <br>

We aim to be able to provide the following, by the end of the time allotted to us:
1. System to conclude the sentiment of real time news pertaining to industries, companies, players, etc, which has an effect on the stock prices.
2. Stock Price Prediction model with optimal accuracy
3. Easy and user friendly interface for a user to take complete advantage of the model developed
4. View of real time news with respect to the stock market (Economic, business, Finance,
Government, Public) <br> 

The above deliverables have been decided upon by keeping the scope of the project in mind. We’ve thought about both the internal and external stakeholders and will ensure the deliverables are a result of deliberate work and effort put in. Each deliverable has a definite role in accomplishing the project’s objective as a whole. <br> 

If time permits, we would like to research and implement the following in our project:
1. Customer Profiling based on investments made, as well as current market situations.
2. Developing a chat bot, for FAQs and specific questions.
3. Integration of demat accounts.
4. Risk Analysis of other investment sectors <br> 

[Back to Top](#table-of-contents)

## Project So Far
### Architecture
(insert image here)
### Implementation

 - Collected and performed data engineering on the **dynamic NIFTY-50 Stock Market Dataset** that had Open, High, Low, Close, and Turnover values of 50 most influential Indian companies from **June 2000 to June 2020** and ongoing. 
 - Trained a **baseline linear regression model** that obtained an **accuracy of 65 %.** 
 - LSTMs neural networks are special types of Recurrent neural networks that have loops in them, allowing information to persist. This means that not only were they able to update the weights of their nodes frequently, but they were also able to take into consideration previous weights, which makes them ideal for time-series data such as stock price data.
 - With a **9-layer neural network,** having 4 50-node LSTM layers followed alternatively by 4 0.2 dropout layers and the last layer being a Dense layer with adam optimizer and mean squared error as the loss function, the model **trained over 100 epochs** to give an **RMSE value of 13.896**. 
 - To incorporate an NLP model to assess twitter or news data we have so far tried to generate a time series dataset of all the news or twitter feed related to each of the companies in the NIFTY-50 dataset and append the news entry of that day with the respective stock price of that company in the dataset. 
 - On this new dataset, we used sentiment analysis models like Bag-of-Words as a baseline and chose to test with **Word2Vec, BERT, NLTIK, and ERNIE** to generate scores on a scale of +10 to -10 with +10 being extremely positive for the stock’s health to -10 being the worst. 
 - The entire system **updates its news scores** when stock prices correlated to a particular type of news reaches new extremes. 
 
 
### Kaggle Notebooks
 Link to Kaggle Notebooks:<br>https://www.kaggle.com/samyakjhaveri/regression-on-stock-data 
 https://www.kaggle.com/samyakjhaveri/stock-prediction-using-lstm-on-tata-global-stock

[Back to Top](#table-of-contents)

## Presentations
Link to Presentation 1: https://docs.google.com/presentation/d/1BeR03mGtyhEVL-2rF2ebzFm_XnP_DQ_bwYUNVk3idio/edit?usp=sharing

[Back to Top](#table-of-contents)
 
## Challenges
 - Preparing a dataset of the Indian stock market on the Nifty-50 index, 
 - Correlate with the corresponding news/twitter feed of that company for that day,
 - **Building a language processing model** that **dynamically generates vector scores** of the news’/twitter feeds’ effect on the stock price and updates the LSTM model accordingly 

[Back to Top](#table-of-contents)

## Further Work
We intend to build a **pipeline for this entire process** to work in flow with real-time daily stock price data and news data as input which would pass from the feature engineering and correlation module to the feature loading model, and finally to the model training and testing modules. The models will be trained and updated on a **GPU accelerated platform**, like Kaggle’s or on Google Colab.

[Back to Top](#table-of-contents)
