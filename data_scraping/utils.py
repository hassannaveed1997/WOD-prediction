# Write any functions here for datascraping, so they can be reused
import requests
import json
from bs4 import BeautifulSoup
from openai import OpenAI


def pull_all_data(division):
    """
    A good start would be this reddit thread, which shows someone else had scraped this data before: https://www.reddit.com/r/crossfit/comments/129kwir/crossfit_games_api/
    """
    base_url = "https://c3po.crossfit.com/api/competitions/v2/competitions/open/2023/leaderboards"
    main_query = f"view=0&division={division}&region=0&scaled=0&sort=0"
    r = requests.get(f"{base_url}?{main_query}")
    data = r.json()
    page_count = data["pagination"]["totalPages"]
    for i in range(1, page_count + 1):
        full_url = f"{base_url}?{main_query}&page={i}"
        r = requests.get(full_url)
        data = r.json()
        for row in data["leaderboardRows"]:
            row_out = [
                row["entrant"]["competitorName"],
                row["scores"][0]["rank"],
                row["scores"][1]["rank"],
                row["scores"][2]["rank"],
                row["scores"][3]["rank"],
            ]
            print(",".join(row_out))
    # TODO: return it in the right format
    return 

def scrape_workout_description(year, workout, division = 1):
    """
    This function will scrape the workout description from the crossfit games website

    Parameters:
    ----------
    year: int
        The year of the open, for example 2023
    workout: int
        The workout number, for example 23.1 return 1
    division: int
        The division of the athlete. Men Rx is division 1. For more details, refer to division mapping

    Returns:
    -------
    description: str
        The description of the workout as a string.
    """
    # url = f"https://games.crossfit.com/competition/open/{year}/workouts/{workout}"
    url = f"https://games.crossfit.com/workouts/open/{year}/{workout}?division={division}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error with year: {year}, workout: {workout}")
        return None
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    description = soup.find_all('div', class_='exercises')
    if len(description) == 0:
        print(f"Error with year: {year}, workout: {workout}")
        return None
    description = description[0].text
    return description

def clean_description(name, description):
    """
    This function will clean the description of the workout, by parsing some important information from it such as the goal and time cap.

    Parameters:
    ----------
    description: str
        The description of the workout as a string.

    Returns:
    -------
    cleaned_description: str
        The cleaned description of the workout as a string.
    """
    prompt = f"""You would be a given a crossfit workout, and will return a json object with the following fields:

    "goal": "The goal of the workout, either 'for time','AMRAP','load'. AMRAP stands for 'as many rounds/reps as possible'",
    "time_cap": "The time cap for the workout in minutes, if any. Otherwise null",
    "total_reps": "The total number of reps in the workout, if the workout is for time, otherwise null",
    "description": "A description of the workout, copy pasted from above, but without goal and timecap lines"

    Some things to keep in mind:
    - Some workouts descriptions may have more than one workout, such as 23.2A and 23.2B, in that case, treat each seperately. The json object should have 2 entries, one named 23.2A and the other 23.2B

    The first two are done for you:

    -----
    17.6:
    Complete as many rounds and reps as possible in 13 minutes of:
    55 deadlifts, 225 lb.
    55 wall-ball shots, 20-lb. ball to 10-ft. target
    55-calorie bike
    55 pullups
    -----
    {{
    "17.6":{{
    "goal": "AMRAP",
    "time_cap": 13,
    "total_reps": null,
    "description": "55 deadlifts (225 lb), 55 wall-ball shots (20-lb ball to 10-ft target), 55-calorie bike, 55 pullups"
    }}
    }}
    -----
    {name}:
    {description}
    -----
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ],
    )
    cleaned_description = completion.choices[0].message.content
    return cleaned_description



