# Project Name
> Capstone Project Submission by Nitin Balaji Srinivasan, Cohort 58 - AI and ML

Sentiment Based Product Recommendation System




## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
Problem Statement and Objective:
Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

Therefore it is required to build a model that will improve the recommendations given to the users given their past reviews and ratings.

We need to build a sentiment-based product recommendation system, which includes the following tasks.

- Data sourcing and sentiment analysis
- Building a recommendation system
- Improving the recommendations using the sentiment analysis model
- Deploying the end-to-end project with a user interface


Approach and High Level Steps:
1. Data sourcing and sentiment analysis
In this task, we have to analyse product reviews after some text preprocessing steps and build a ML model to get the sentiments corresponding to the users' reviews and ratings for multiple products.

The Product Reviews Dataset consists of 30,000 reviews for more than 200 different products. The reviews and ratings are given by more than 20,000 users.

The steps to be performed for the first task are given below.

    1) Exploratory data analysis

    2) Data cleaning

    3) Text preprocessing

    4) Feature extraction: In order to extract features from the text data, we may choose from any of the methods, including bag-of-words, TF-IDF vectorization or word embedding.

    5) Training a text classification model: We need to build at least three ML models. We need to analyse the performance of each of these models and choose the best model. At least three out of the following four models need to be built (incl.handling the class imbalance and hyperparameter tuning if required). 
        1. Logistic regression
        2. Random forest
        3. XGBoost
        4. Naive Bayes

        Out of these four models, we need to select one classification model based on its performance.

2. Building a recommendation system
We can use the following types of recommendation systems.

- User-based recommendation system

- Item-based recommendation system

We should analyse the recommendation systems and select the one that is best suited in this case.

Once we get the best-suited recommendation system, the next task is to recommend 20 products that a user is most likely to purchase based on the ratings. We can use the 'reviews_username' (one of the columns in the dataset) to identify the user.

3. Improving the recommendations using the sentiment analysis model
Now, the next task is to link this recommendation system with the sentiment analysis model that was built earlier (recall that we asked you to select one ML model out of the four options). Once we recommend 20 products to a particular user using the recommendation engine, we need to filter out the 5 best products based on the sentiments of the 20 recommended product reviews.

In this way, we will get an ML model (for sentiments) and the best-suited recommendation system.

4. Deployment of this end to end project with a user interface
Once we get the ML model and the best-suited recommendation system, we will deploy the end-to-end project. We need to use the Flask framework, which is majorly used to create web applications to deploy machine learning models.

To make the web application public, we need to use Heroku, which works as the platform as a service (PaaS) that helps developers build, run and operate applications entirely on the cloud.

Next, we need to include the following features in the user interface.

- Take any of the existing usernames as input.
- Create a submit button to submit the username.
- Once you press the submit button, it should recommend 5 products based on the entered username.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
We have developed 5 text classification models for sentiment analysis
 - Logistic Regression
 - Decision Tree
 - Random Forest
 - Naive Bayes
 - XGBoost

 Logistic Regression provided the best performance with a high level of ROC_AUC_Score and Accuracy>95%.
 Therefore this model was chosen for the sentiment analysis.

 Further we developed, user-user based recommendation system and item-item based recommendation system. Upon evaluation of RMSE, user-user recommendation system performed better.

 Using user-user based recommendation and Logistic Regression, we generated top 5 recommendations for every user. 

 Furthermore, we deployed the model into an app, that is publically visible

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Pandas
- Numpy
- matplotlib
- seaborn
- NLTK
- Scikit learn

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->


## Contact
Created by [@nbsrini] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->