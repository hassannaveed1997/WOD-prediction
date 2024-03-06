import json
import difflib
import random
from openai import OpenAI

client = OpenAI()

with open("../../Data/assets/movement_descriptions.json", "r") as file:
    movements = json.load(file)

def get_movement_description(movement):
    if movement not in movements:
        # find most similar movement
        closest_match = difflib.get_close_matches(movement, movements.keys(), n=1, cutoff=0.5)
        if closest_match:
            return movements[closest_match[0]]
        else:
            raise ValueError(f"Movement {movement} not found in the movement_descriptions.json")

    return movements[movement]

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding