#!/bin/bash

# python main.py datasets/test.csv samples label output/test.csv
python main.py datasets/Jigsaw_Toxicity.csv ChatGPT comment_text toxicity output/Jigsaw_Toxicity.csv -ic id -ic entropy
python main.py datasets/CoLA.csv ChatGPT Text Label output/CoLA.csv -sd "the grammatic acceptability of" -de
python main.py datasets/ColBERT.csv ChatGPT 'text' 'humor' output/ColBERT.csv
python main.py datasets/iSarcasm.csv ChatGPT 'tweet' 'sarcastic' output/iSarcasm.csv
python main.py datasets/IMDBSentiment.csv ChatGPT Text Label output/IMDBSentiment.csv -sd "the sentiment of"
python main.py datasets/MathQA.csv ChatGPT question answer output/MathQA.csv --task PROBLEM_SOLVER
python main.py datasets/SuperGlueNLI.csv ChatGPT Sample Label output/SuperGlueNLI.csv
python main.py datasets/CoPA.csv ChatGPT sample label output/CoPA.csv --task PLAUSIBLE_ALTERNATIVE
python main.py datasets/BoolQA.csv ChatGPT sample label output/BoolQA.csv --task TF_PASSAGE
python main.py datasets/TweetStance.csv ChatGPT Sample Stance output/TweetStance.csv --task SIMPLE_STANCE
python main.py datasets/ReAD.csv ChatGPT Samples Labels output/ReAD.csv --task PASSAGE_CHOICE