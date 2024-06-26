"""
Utility Program: scorer.py
Author: Devon McDermott
Date: 3/13/2024

Introduction:
This utility program compares the sense tagged output generated by a Word Sense Disambiguation (WSD) system with the key
 data to evaluate the accuracy of the tagging and provide a confusion matrix.

Input:
- Sense tagged output file generated by the WSD system (my-line-answers.txt)
Here is a snippet of my-line-answers.txt:
<answer instance="line-n.w8_053:3883:" senseid="product"/>
<answer instance="line-n.w8_040:16402:" senseid="product"/>
<answer instance="line-n.w7_065:13727:" senseid="product"/>
<answer instance="line-n.w8_007:14740:" senseid="product"/>

- Key data file containing the correct sense tags (line-key.txt)
Here is a snippet of line-key.txt:
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>
<answer instance="line-n.w9_40:10187:" senseid="phone"/>

Output:
- Overall accuracy of the tagging
Here is an example of how this looks:
Overall Accuracy: 50.79365079365079

- Confusion matrix showing the distribution of predicted senses compared to the true senses
Here is an example of the confusion matrix:
------------------------------
phone     48        24
product   40        14


Algorithm:
1. Read the sense tagged output file and key data file.
2. Extract instance IDs and predicted sense IDs from the sense tagged output file.
3. Extract instance IDs and true sense IDs from the key data file.
4. Calculate the overall accuracy by comparing predicted sense IDs with true sense IDs.
5. Build a confusion matrix to visualize the distribution of predicted senses compared to true senses.
6. Print the overall accuracy and confusion matrix to STDOUT.

INSTRUCTIONS FOR USE:
****USE AFTER RUNNING WSD.PY or wsd-ml.py
1. Open the terminal on your computer
2. Navigate to the directory with the relevant files
example commands: cd wsd 
3. type this command into your terminal:
python3 scorer.py my-line-answers.txt line-key.txt >> my-line-answers.txt
4. Open my-line-answers.txt and scroll to the bottom to see the appended accuracy and confusion matrix

**********************************
BE SURE TO USE ">> my-line-answers.txt" at the end of the command. Using one ">" will overwrite the file
using two ">>" will simply append the my-line-answers.txt which is what we want here.
"""

import sys
from collections import defaultdict

def read_answers(answers_file):

    #Read the sense tagged output file and extract instance IDs and predicted sense IDs.

    answers = {}
    with open(answers_file, 'r') as file:
        for line in file:
            instance_id, sense_id = extract_instance_and_sense(line)
            answers[instance_id] = sense_id
    return answers

def read_key(key_file):

    #Read the key data file and extract instance IDs and true sense IDs.

    key = {}
    with open(key_file, 'r') as file:
        for line in file:
            instance_id, sense_id = extract_instance_and_sense(line)
            key[instance_id] = sense_id
    return key

def extract_instance_and_sense(line):

    #Extract instance ID and sense ID from a line of input.

    parts = line.strip().split('"')
    if len(parts) < 4:
        return None, None
    instance_id = parts[1]
    sense_id = parts[3]
    return instance_id, sense_id

def calculate_accuracy(answers, key):

    #Calculate the overall accuracy of the tagging.

    correct = 0
    total = len(key)
    for instance_id, sense_id in key.items():
        if instance_id in answers and answers[instance_id] == sense_id:
            correct += 1
    accuracy = correct / total * 100
    return accuracy

def build_confusion_matrix(answers, key):

    #Build a confusion matrix to visualize the distribution of predicted senses.

    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for instance_id, sense_id in key.items():
        predicted_sense = answers.get(instance_id, 'unknown')
        confusion_matrix[sense_id][predicted_sense] += 1
    return confusion_matrix

def print_confusion_matrix(confusion_matrix):

    #Print the confusion matrix to STDOUT.

    print("Confusion Matrix:")
    print("{:<10} {:<10} {:<10}".format("", "Predicted", ""))
    print("{:<10} {:<10} {:<10}".format("", "Phone", "Product"))
    print("-" * 30)
    for true_label in confusion_matrix:
        print("{:<10}".format(true_label), end="")
        for predicted_label in confusion_matrix[true_label]:
            print("{:<10}".format(confusion_matrix[true_label][predicted_label]), end="")
        print()

    #MAIN
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scorer.py my-line-answers.txt line-key.txt")
        sys.exit(1)

    answers_file = sys.argv[1]
    key_file = sys.argv[2]

    answers = read_answers(answers_file)
    key = read_key(key_file)

    accuracy = calculate_accuracy(answers, key)
    confusion_matrix = build_confusion_matrix(answers, key)

    print("Overall Accuracy:", accuracy)
    print_confusion_matrix(confusion_matrix)
