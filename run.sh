#!/bin/bash

# python main.py datasets/test.csv samples label output/test.csv
python main.py datasets/Jigsaw_Toxicity.csv comment_text toxicity output/Jigsaw_Toxicity.csv -ic id -ic entropy
