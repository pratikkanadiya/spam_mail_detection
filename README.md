# Spam Mail Detection using Machine Learning
A machine learning project that classifies email/SMS messages as spam or ham (not spam) using Logistic Regression with TF-IDF feature extraction.


 Project Overview

 This project builds a text classification pipeline to detect spam messages. Given a raw email or SMS message, the model predicts whether it is spam or legitimate (ham). The pipeline is trained on a labeled dataset and saved as a deployable .pkl model file.


Project Structure

spam-mail-detection/

   ├── tamplates
      ├── index.html         #frontend code  

   ├── app.py                #Flask web application

   ├── spam_mail.ipynb       # Main Jupyter notebook
   
   ├── mail_data.csv         # Dataset (5572 labeled messages)
   
   ├── spammodel.pkl         # Trained Logistic Regression model
   
   ├── vectorizer.pkl        # Fitted TF-IDF Vectorizer

Results

Training Accuracy: 96.77%

Test Accuracy: 96.68%
