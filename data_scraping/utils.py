# Write any functions here for datascraping, so they can be reused
import requests
import json


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

