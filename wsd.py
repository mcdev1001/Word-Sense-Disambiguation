"""
Word Sense Disambiguation (WSD) Program: wsd.py
Author: Devon McDermott
Date: 3/13/2024

Introduction:
This program implements a Decision List classifier to perform word sense disambiguation.
The goal of WSD is to determine the correct sense of an ambiguous word in context.
The program learns a model from training data and applies that model to test data to assign a sense to the target word.
The algorithm used is based on a bag-of-words feature representation and Decision List classification.

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

- Model output file (my-model.txt) (example below)


Output:
- Predicted sense IDs for each instance in the test data, printed to STDOUT in the key format
Here is an example of one line:
<answer instance="line-n.w9_1:4358:" senseid="product"/>

- Model output file containing learned features, log-likelihoods, and predicted senses (my-model.txt)
Here is what my model output looked like (snippet):
sharply	4.59511985013459	product
pricing	4.59511985013459	product
access	4.59511985013459	product
according	4.59511985013459	product
rocky	4.59511985013459	product
risen	4.59511985013459	product
lately	4.59511985013459

***If we put in the incorrect # of arguments at command line this will print:
Usage:
python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt

Algorithm:
1. Parsing Training Data: Extract context, target word, and sense ID from training data using custom parsing functions.
2. Parsing Test Data: Extract context and target word from test data using custom parsing functions.We need to extract instanceID here.
3. Preprocessing Text: Remove stopwords, punctuation, HTML tags, special characters, and numeric characters from context text.
4. Extracting Bag-of-Words Features: Represent occurrences of words in context as bag-of-words features.
5. Training the Model: Train a Decision List classifier using bag-of-words features. Calculate log-likelihoods for each feature and sense, and sort the model based on log-likelihoods.
6. Saving the Model: Save the trained model to my-model.txt.
7. Classifying Test Data: Classify word sense for each test instance using the trained model. Extract bag-of-words features from context text, and assign the sense associated with the highest log-likelihood.
8. Scoring the Model(in scorer.py): Evaluate the performance of the classifier using the scorer.py utility to calculate accuracy and provide a confusion matrix.

Decision List Details:
The Decision List classifier is trained using bag-of-words features extracted from the training data.
In bag-of-words representation, each feature represents a word in the context of the target word
whose sense needs to be disambiguated.

For each feature, the log-likelihoods are calculated for each sense.
The log-likelihood is a measure of how likely it is for a particular sense to occur given the presence
of a specific feature. It's calculated based on the frequency of occurrence of that feature in instances
labeled with each sense.

The model is then sorted based on log-likelihoods.
 sorting is done in descending order,
so the most likely sense for each feature appears at the top of the model.

Once the model is sorted, each feature is associated with the sense that has the highest log-likelihood
for that feature. This means that when classifying new data the Decision List classifier
looks at the features in the instance and assigns the sense associated with the highest log-likelihood.

Accuracy and Confusion Matrix:
Overall Accuracy: 50.79%
Confusion Matrix:
           Predicted
           Phone      Product
------------------------------
Phone      48         24
Product    40         14

Using this interpretation:
The row labels represent the true senses of the word
The column labels represent the predicted senses of the word
The count of instances where the true sense is "phone" and the predicted sense is also "phone" is 48 (True Positives for "phone").
The count of instances where the true sense is "product" and the predicted sense is "phone" is 40 (False Positives for "phone").
The count of instances where the true sense is "phone" and the predicted sense is "product" is 24 (False Negatives for "phone").
The count of instances where the true sense is "product" and the predicted sense is also "product" is 14 (True Positives for "product").

Comparison to Most Frequent Sense Baseline:
The most frequent sense baseline predicts the most common sense for every instance.
If we assume "phone" is the most frequent sense, the baseline accuracy would be 50%.
The Decision List classifier achieves an accuracy slightly above the baseline, indicating some level of effectiveness.

INSTRUCTIONS FOR USE:
****USE BEFORE RUNNING SCORER.PY
1. Open the terminal on your computer
2. Navigate to the directory with the relevant files
example commands: cd wsd
3. type this command into your terminal:
python3 WSD.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
4. Open my-line-answers.txt and see STDOUT

"""
import sys
import re
from collections import defaultdict
import math
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stopwords


def custom_parse_instance(instance):
    #parser built custom for our pseudo xml data to extract context, target word, and sense ID

    # Extract context between <context> tags
    context_match = re.search(r'<context>(.*?)</context>', instance, re.DOTALL)
    if context_match:
        context = context_match.group(1)
        # Remove <s> and </s> tags
        context = re.sub(r'<s>|</s>', '', context)
        # Extract text between <head> tags
        target_match = re.search(r'<head>(.*?)</head>', context)
        if target_match:
            target_word = target_match.group(1)
            # Extract senseid from <answer> tag
            sense_match = re.search(r'<answer.*?senseid="(.*?)"/>', instance)
            if sense_match:
                sense_id = sense_match.group(1)
            else:
                sense_id = None
        else:
            target_word = None
            sense_id = None
        # Remove leading and trailing whitespace
        context = context.strip()
        return context, target_word, sense_id
    else:
        return None, None, None

def preprocess_text(text):
    #Preprocesses text by removing stopwords and punctuation
    text = text.lower()                 # Convert to lowercase
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'@\s*', ' ', text)   # Remove special characters
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\d', '', text)  # Remove numeric
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]  # Remove stopwords
    return ' '.join(filtered_words)

def parse_training_data(filename):
    #Parse training data
    training_data = []

    with open(filename, 'r') as file:
        data = file.read()
        instances = data.split('<instance ')[1:]  # Skip the first element (empty string)

        for instance in instances:
            context, target_word, sense_id = custom_parse_instance('<instance ' + instance)
            if context and target_word and sense_id:
                # Preprocess the context
                context = preprocess_text(context)
                training_data.append((context, target_word, sense_id))


    return training_data

def parse_testing_data(test_filename):
    #Parse testing data (seprately from training data)
    testing_data = []

    with open(test_filename, 'r') as file:
        data = file.read()
        instances = data.split('<instance ')[1:]  # Skip the first element (empty string)

        for instance in instances:
            instance_id_match = re.search(r'id="(.*?)"', instance)
            if instance_id_match:
                instance_id = instance_id_match.group(1)
                context, target_word, _ = custom_parse_instance('<instance ' + instance)
                if context and target_word:
                    # Preprocess the context
                    context = preprocess_text(context)
                    testing_data.append((instance_id, context, target_word))

    return testing_data


def train_model(training_data):
    """
    Train a Decision List model from training data.
    Return a list of (feature, log-likelihood, sense) tuples.
    """
    # Count occurrences of each feature for each sense
    feature_count = defaultdict(lambda: defaultdict(int))
    sense_count = defaultdict(int)
    for sentence, _, sense_id in training_data:
        features = extract_features(sentence)
        sense_count[sense_id] += 1
        for feature, count in features.items():
            feature_count[feature][sense_id] += count

    total_senses = len(sense_count)
    model = []

    # Calculate log-likelihood for each feature and sense
    for feature, counts in feature_count.items():
        for sense_id, count in counts.items():
            likelihood = abs(math.log((count + 1) / (sense_count[sense_id] + total_senses)))
            model.append((feature, likelihood, sense_id))

    # Sort the model based on log-likelihoods
    model.sort(key=lambda x: x[1], reverse=True)
    return model

def extract_features(sentence):
    """
    Extract bag-of-words features from the sentence
    Return a dictionary where keys are words and values are their counts.
    """
    features = defaultdict(int)
    words = sentence.split()  # Extracted words are already preprocessed
    for word in words:
        features[word] += 1
    return features

def classify_sentence(instance_id, sentence, model):
    """
    Classify the given sentence using Decision List model
    Return the predicted sense
    write the predictions in SAME FORMAT as key format with stdout
    """

    features = extract_features(sentence)
    max_score = -float('inf')
    predicted_sense = None
    for feature, likelihood, sense_id in model:
        if feature in features:
            score = likelihood * features[feature]
            if score > max_score:
                max_score = score
                predicted_sense = sense_id

    sys.stdout.write(f"<answer instance=\"{instance_id}\" senseid=\"{predicted_sense}\"/>\n")
    return predicted_sense

def save_model(model, model_filename):

    #Save the learned model

    with open(model_filename, 'w') as file:
        for feature, likelihood, sense_id in model:
            file.write(f"{feature}\t{likelihood}\t{sense_id}\n")


#MAIN
def main(train_filename, test_filename, model_filename):
    # Read training and test data
    training_data = parse_training_data(train_filename)
    test_data = parse_testing_data(test_filename)

    # Train the model
    model = train_model(training_data)

    # Save the model
    save_model(model, model_filename)

    # Classify test data
    for instance_id, sentence, target_word in test_data:
        predicted_sense = classify_sentence(instance_id, sentence, model)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
