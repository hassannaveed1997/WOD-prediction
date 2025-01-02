import json

import requests
from bs4 import BeautifulSoup


def retrieve_benchmark_info(athlete_id):
    """
    This function will scrape the benchmark statistics for an athlete.

    Parameters:
    ----------
    athlete_id: an id assigned to identify an athlete

    Returns:
    -------
    description: a list of all the available bench mark statistics of an athlete. Will only
    return an exercises' corresponding value if present.
    """
    url = f"https://games.crossfit.com/athlete/{athlete_id}"

    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, "lxml")

    try:
        name_container = soup.find("div", class_="athlete-name").find_all("span")
        name = name_container[0].text + " " + name_container[1].text
        benchmark_stats_container = soup.find("div", id="benchmarkStats").find(
            "ul", class_="stats-container"
        )
    except AttributeError:
        return f"An error occurred when trying to retrieve athlete id: {athlete_id} benchmark statistics."
    else:
        dict_ = {"athlete_id": athlete_id, "name": name}
        stats_section = benchmark_stats_container.find_all(
            "div", class_="stats-section"
        )
        for section in stats_section:
            values_list = section.find_all("td")
            exercises_list = section.find_all("th", class_="stats-header")
            for value_html, exercise_html in zip(values_list, exercises_list):
                value = value_html.text.strip()
                exercise = exercise_html.text.strip()
                if value != "--":
                    dict_[exercise] = value
    return dict_
