from openai import OpenAI
import os
import pandas as pd
import csv
import json
import ast
from extract_functions import get_overall_difficulty, extract_levels, extract_features

with open('open_parsed_descriptions.json', 'r') as f:
    descriptions = json.load(f)

descriptions_list = []
levels = []

for key in descriptions:
    descriptions_list.append(descriptions[key]["description"])
    levels.append(ast.literal_eval(extract_levels(descriptions[key]["description"])))

difficulties = []
for i in range(len(levels)):
    difficulties.append(get_overall_difficulty(levels[i][0],levels[i][1],levels[i][2],levels[i][3],levels[i][4]))

features = []
for i in range(len(descriptions_list)):
    features.append(extract_features(descriptions_list[i],difficulties[i]))

headers = "Repetition, Time, Format, Equipment, ExerciseType, TargetMuscle, Difficulty"
headers = headers.split(", ")
data = [item.split(",") for item in features]

with open("features_extracted1.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers
    writer.writerows(data)    # Write data rows