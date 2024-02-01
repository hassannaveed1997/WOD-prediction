from urllib.request import urlopen 
import pandas as pd
import json

athlete_info_data = []
scores_data = []

i = 1

x= -i
y= 0
for j in range(3390):
    x += i
    y += i
    j += 1
    print(j)
    for num in range(x, y):
        url = f'https://c3po.crossfit.com/api/leaderboards/v2/competitions/open/2023/leaderboards?view=0&division=1&region=0&scaled=0&sort=0&page={num}'
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
                row["scores"][0]["scoreDisplay"],
                row["scores"][1]["scoreDisplay"],
                row["scores"][2]["scoreDisplay"],
                row["scores"][3]["scoreDisplay"],
            ]
            scores_data.append(scores_entry)

# Create DataFrames
athlete_info_df = pd.DataFrame(athlete_info_data, columns=["id", "name", "age", "height", "weight"])
scores_df = pd.DataFrame(scores_data, columns=["id", "23.1", "23.2A", "23.2B", "23.3"])
print(athlete_info_df)
print(scores_df)

athlete_info_df.to_csv('2023_Men_info.csv')
scores_df.to_csv('2023_Men_scores.csv')
