#!/bin/bash

# python main.py datasets/test.csv samples label output/test.csv
python main.py datasets/Jigsaw_Toxicity.csv comment_text toxicity output/Jigsaw_Toxicity.csv -ic id -ic entropy
python main.py datasets/CoLA.csv ChatGPT Text Label output/CoLA.csv -sd "the grammatic acceptability of" -de
python main.py datasets/ColBERT.csv ChatGPT 'text' 'humor' output/ColBERT.csv