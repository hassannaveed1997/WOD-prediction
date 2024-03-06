from urllib.request import urlopen 
import pandas as pd
import json
import sys

with open("../../Data/assets/year_to_column_mapping.json", "r") as f:
    year_to_col_name_mapping = json.loads(f.read())

with open("../../Data/assets/division_mapping.json", "r") as f:
    division_mapping = json.loads(f.read())

def scrape_data_for_year(year, division = "mens",num_pages = None):
    athlete_info_data = []
    scores_data = []

    division_num = division_mapping[division.lower()]

    if str(year) not in year_to_col_name_mapping:
        raise ValueError(f"Year {year} not in year_to_col_name_mapping, please check")

    # get number of pages
    url = f'https://c3po.crossfit.com/api/leaderboards/v2/competitions/open/{year}/leaderboards?view=0&division={division_num}&region=0&scaled=0&sort=0&page=1'
    response = urlopen(url)
    data = json.loads(response.read())
    if num_pages is None:
        num_pages = data['pagination']['totalPages']
    num_workouts = len(data['leaderboardRows'][0]['scores'])

    for i in range(1,num_pages+1):
        url = f'https://c3po.crossfit.com/api/leaderboards/v2/competitions/open/{year}/leaderboards?view=0&division={division_num}&region=0&scaled=0&sort=0&page={i}'
        response = urlopen(url)
        api_response = json.loads(response.read())
        for row in api_response["leaderboardRows"]:
            athlete_info_data.append([
                row["entrant"]["competitorId"],
                row["entrant"]["competitorName"],
                row["entrant"]["age"],
                row["entrant"]["height"],
                row["entrant"]["weight"],
            ])
            scores_entry = [
                row["entrant"]["competitorId"],
            ] + [row["scores"][j]["scoreDisplay"] for j in range(num_workouts)]

            scores_data.append(scores_entry)
            
        # print every 10th page to check progress
        if i % 10 == 0:
            print(f"Finished page {i} of {num_pages}")

    # Create DataFrames
    athlete_info_df = pd.DataFrame(athlete_info_data, columns=["id", "name", "age", "height", "weight"])
    scores_df = pd.DataFrame(scores_data, columns=["id"]+year_to_col_name_mapping[str(year)])

    # save to csv
    athlete_info_df.to_csv(f'../../Data/{year}_{division}_info.csv')
    scores_df.to_csv(f'../../Data/{year}_{division}_scores.csv')

    return athlete_info_df, scores_df
