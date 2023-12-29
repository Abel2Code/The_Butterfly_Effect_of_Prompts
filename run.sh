#!/bin/bash

python main.py datasets/Jigsaw_Toxicity.csv ChatGPT comment_text toxicity -ic id -ic entropy
python main.py datasets/CoLA.csv ChatGPT Text Label -sd "the grammatic acceptability of" -de
python main.py datasets/ColBERT.csv ChatGPT 'text' 'humor'
python main.py datasets/iSarcasm.csv ChatGPT 'tweet' 'sarcastic'
python main.py datasets/IMDBSentiment.csv ChatGPT Text Label -sd "the sentiment of"
python main.py datasets/MathQA.csv ChatGPT question answer --task PROBLEM_SOLVER
python main.py datasets/SuperGlueNLI.csv ChatGPT Sample Label 
python main.py datasets/CoPA.csv ChatGPT sample label --task PLAUSIBLE_ALTERNATIVE
python main.py datasets/BoolQA.csv ChatGPT sample label --task TF_PASSAGE
python main.py datasets/TweetStance.csv ChatGPT Sample Stance --task SIMPLE_STANCE
python main.py datasets/ReAD.csv ChatGPT Samples Labels --task PASSAGE_CHOICE

python main.py datasets/Jigsaw_Toxicity.csv LLaMA-70B comment_text toxicity -ic id -ic entropy
python main.py datasets/CoLA.csv LLaMA-70B Text Label -sd "the grammatic acceptability of" -de
python main.py datasets/ColBERT.csv LLaMA-70B 'text' 'humor'
python main.py datasets/iSarcasm.csv LLaMA-70B 'tweet' 'sarcastic'
python main.py datasets/IMDBSentiment.csv LLaMA-70B Text Label -sd "the sentiment of"
python main.py datasets/MathQA.csv LLaMA-70B question answer --task PROBLEM_SOLVER
python main.py datasets/SuperGlueNLI.csv LLaMA-70B Sample Label 
python main.py datasets/CoPA.csv LLaMA-70B sample label --task PLAUSIBLE_ALTERNATIVE
python main.py datasets/BoolQA.csv LLaMA-70B sample label --task TF_PASSAGE
python main.py datasets/TweetStance.csv LLaMA-70B Sample Stance --task SIMPLE_STANCE
python main.py datasets/ReAD.csv LLaMA-70B Samples Labels --task PASSAGE_CHOICE