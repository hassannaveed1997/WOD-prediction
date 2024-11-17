from openai import OpenAI
import os
import pandas as pd
import csv
import json
import ast

client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))

def ChatGPT_API(prompt):
  # Create a request to the chat completions endpoint
  response = client.chat.completions.create(
    model="gpt-4o",
    # Assign the role and content for the message
    messages=[{"role": "user", "content": prompt}], 
    temperature = 0) # Temperature: Controls the randomness of the response (0 is deterministic, 2 is highly random).
  return response.choices[0].message.content

level_map = {'high':2.5, 'medium': 2,'low':1.5}
weights = [0.3, 0.25, 0.15, 0.2, 0.1]

   
def get_numeric_level(level):
   return level_map[level]

#passed in values here should be high, medium, or low
def get_overall_difficulty(aerobic_level,intensity,stability_demand,explosiveness,weight_class):
  values = [
    get_numeric_level(aerobic_level),
    get_numeric_level(intensity),
    get_numeric_level(stability_demand),
    get_numeric_level(explosiveness),
    get_numeric_level(weight_class)
  ]
  overall_difficulty = sum(v * w for v, w in zip(values, weights))
  if overall_difficulty<2:
    return 'low'
  elif overall_difficulty<2.1:
    return 'medium'
  else:
    return 'high'
  
def extract_levels(description):
    prompt = (
        f"Given the following crossfit workout description: \n"
        f"{description}\n\n"
        f"Please extract the following features:\n"
        f"1. AerobicLevel - high, medium, low\n"
        f"2. Intensity - high, medium, low\n"
        f"3. StabilityDemand - high, medium, low\n"
        f"4. Explosiveness - high, medium, low\n"
        f"5. WeightClass - high, medium, low\n\n"
        f"Provide the categorizations of the features as a list. Keep in mind that these are crossfit workouts, meaning you have to assess these features based off of their relative difficulty compared to general crossfit workouts, meaning they are comarably SIGNIFICANTLY less intense than you think. In essence, make sure to underestimate." 
        f"For example the list should look something like ['high','medium','low','low','high'] for each respective feature.\n"
        f"Make sure you only output the list of feature and nothing else."
    )

    features = ChatGPT_API(prompt)
    return features


def extract_features(description,difficulty):
    # Format the prompt to extract the specified features
    prompt = (
        f"Given the following workout description:\n\n"
        f"{description}\n\n"
        f"Please extract the following features:\n"
        f"1. Repetition - The amount of repetitions in total\n"
        f"2. Time - The time that you should take to do the exercise\n"
        f"3. Format - AMRAP, EMOM, or rounds for time\n"
        f"4. Equipment - Type of equipment example: barbell, dumbbell, squat rack, machine, etc.\n"
        f"5. ExerciseType - The type of exercises performed\n"
        f"6. TargetMuscle - The muscle group targeted as a result of the workout\n"
        f"7. Difficulty - {difficulty}\n"
        f"Provide the extracted features in the description in a common seperated format, with each feature in the format being seperated by commas. ONLY give me the comma seperated features in your response, nothing else, not even the (`) character or the word (csv). For the repetition feature, make sure to give me a number if the workout is not mentioning anything to do with doing as many reps as possible (AMRAP). IF the workout if AMRAP, then make the repetition feature -1. If you cannot find a feature from the description, then just label that feature as undefined in the output you give me. Make sure to only use commas whenever we want to seperate the features. If there requires a list of attributes within a feature (such as listing multiple types of equipment in the Equipment category) then just use the word and."
    )
    
    # Call the ChatGPT API with the formatted prompt
    features = ChatGPT_API(prompt)
    return features