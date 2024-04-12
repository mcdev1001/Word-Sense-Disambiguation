"""
Word Sense Disambiguation/Machine Learning Program: wsd-ml.py
Author: Devon McDermott
Date: 3/25/2024

Introduction:
This program, wsd-ml.py, utilizes 3 machine learning models: Naive Bayes, Logistic Regression, and Support Vector Machine (SVM) to perform Word Sense Disambiguation (WSD). 
Given a set of training data containing instances of an ambiguous word "line" used in different senses, the program learns to predict the correct sense of the word in new sentences from a test corpus. 
It uses bag-of-words feature representation to extract features and applies the trained models to classify the senses of the ambiguous word 

Input:
- Training data file (line-train.txt) Here is a snippet:
<instance id="line-n.w9_10:6830:">
<answer instance="line-n.w9_10:6830:" senseid="phone"/>
<context>
 <s> The New York plan froze basic rates, offered no protection to Nynex against an economic downturn that sharply cut demand and didn't offer flexible pricing. </s> <@> <s> In contrast, the California economy is booming, with 4.5% access <head>line</head> growth in the past year. </s>
</context>
</instance>

- Test data file (line-test.txt) Here is a snippet:
<instance id="line-n.w8_059:8174:">
<context>
 <s> Advanced Micro Devices Inc., Sunnyvale, Calif., and Siemens AG of West Germany said they agreed to jointly develop, manufacture and market microchips for data communications and telecommunications with an emphasis on the integrated services digital network. </s> <@> </p> <@> <p> <@> <s> The integrated services digital network, or ISDN, is an international standard used to transmit voice, data, graphics and video images over telephone <head>lines</head> . </s>
</context>
</instance>


Output:
- Predicted sense IDs for each instance in the test data, printed to STDOUT in the key format
Here is an example of one line:
<answer instance="line-n.w9_1:4358:" senseid="product"/>



***If we put in the incorrect # of arguments at command line this will print:
Usage:
python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt

Algorithm:
1. Parsing Training Data: Extract context, target word, and sense ID from training data using custom parsing functions
2. Parsing Test Data: Extract context and target word from test data using custom parsing functions.We need to extract instanceID here
3. Preprocessing Text: Remove stopwords, punctuation, HTML tags, special characters, and numeric characters from context text using scikit learn stopword list
4. Extracting Bag-of-Words Features: Represent occurrences of words in context as bag-of-words features
5. Training the Model: Here we use Naive Bayes, SVM, and Logistic Regression from scikit learn
6. Classifying Test Data: Classify word sense for each test instance using the trained model 
7. Scoring each Model(in scorer.py): Evaluate the performance of each classifier using the scorer.py utility to calculate accuracy and provide a confusion matrix.

Description of Models:
SVM (Support Vector Machine): A supervised learning algorithm that separates classes by finding the hyperplane that maximizes the margin between them.
In WSD: SVM is utilized to classify the senses of ambiguous words by mapping the feature vectors into a high-dimensional space. It aims to find the optimal hyperplane that separates the instances belonging to different sense classes with the maximum margin
SVM handles decision boundaries and works well with high-dimensional feature spaces

Naive Bayes: A probabilistic classifier based on Bayes theorem with independence assumptions between features
In WSD: Naive Bayes is used to classify the senses of ambiguous words based on the bag-of-words representation. It calculates the likelihood of each sense given the features and selects the sense with the highest probability.
Naive Bayes is performs well with high-dimensional data such as word features from a corpus.

Logistic Regression: A model for binary classification that predicts the probability of an instance belonging to a particular class.
In WSD: Logistic Regression is applied to predict the sense of an ambiguous word based on the features extracted. It estimates the probability of each sense class and the sense with the highest probability is the prediction.
Logistic Regression estimates based on past occurrences. It is simple and works when the data can be separated into two groups

 Accuracy and Confusion Matricies:
 (The rows represent true occurrances and columns predicted)

 SVM:
 Overall Accuracy: 89.68%
 Confusion Matrix:
           Predicted            
           Phone      Product   
 ------------------------------
 phone     7         65        
 product   48        6        

 Naive Bayes:
 Overall Accuracy: 92.86%
 Confusion Matrix:
           Predicted            
           Phone      Product   
 ------------------------------
 phone     67        5         
 product   50        4        

 Logistic Regression:
 Overall Accuracy: 93.65%
 Confusion Matrix:
           Predicted            
           Phone      Product   
 ------------------------------
 phone     67        5         
 product   51        3       

 Comparison to Most Frequent Sense Baseline:
 The most frequent sense baseline predicts the most common sense for every instance.
 If we assume "phone" is the most frequent sense, the baseline accuracy would be around 50%.
 All three models (SVM, Naive Bayes, and Logistic Regression) achieve accuracies significantly higher than the baseline, 
 meaning they are distinguishing between different senses of the ambiguous word.

Decision list model results: 

Accuracy and Confusion Matrix:
Overall Accuracy: 50.79%
Confusion Matrix:
           Predicted
           Phone      Product
------------------------------
Phone      48         24
Product    40         14

Comparison to Most Frequent Sense Baseline:
The most frequent sense baseline predicts the most common sense for every instance.
If we assume "phone" is the most frequent sense, the baseline accuracy would be 50%.
The Decision List classifier achieves an accuracy slightly above the baseline, indicating some level of effectiveness.


COMPARING THE 3 MODELS TO THE DECISION LIST CLASSIFIER
(Most Frequent Sense Baseline):

The Decision List model achieves an overall accuracy of 50.79% where as
the SVM, Naive Bayes, and Logistic Regression models demonstrate higher overall 
accuracies of 89.68%, 92.86%, and 93.65%, respectively. 
The results indicate that the SVM, Naive Bayes, and Logistic Regression models 
outperform the Decision List model in distinguishing between different senses of the ambiguous word.


INSTRUCTIONS FOR USE:
****USE BEFORE RUNNING SCORER.PY

1. Open the terminal on your computer

2. Navigate to the directory with the relevant files
example commands: cd wsd-ml

3. type this command into your terminal:
python3 wsd-ml.py line-train.txt line-test.txt [OPTIONAL: ml-model] > my-line-answers.txt

***NOTE: CASE SENSITIVE!! replace [OPTIONAL: ml-model] with either "SVM", "NaiveBayes", or "LogisticRegression"
***NOTE: will return an error if you use lowercase:
ValueError: Invalid model type. Please choose from 'NaiveBayes', 'LogisticRegression', or 'SVM
***NOTE: will default to NaiveBayes
4. Open my-line-answers.txt and see STDOUT

5. PROCEED TO SCORER.PY



"""
