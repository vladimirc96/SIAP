import requests
from bs4 import BeautifulSoup
import os
import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# id wrappera = voting_results_DIVIZIJA
# svaka tabela ima id formata: voting-results-DIVIZIJA_IDPOZICIJE
# id za pozocije: 1 - Guard, 2 - Forward, 3 - Center
def tableDataText(table):
    rows = []
    trs = table.find_all('tr')
    headerow = [td.get_text(strip=True) for td in trs[0].find_all('th')] # header row
    if headerow: # if there is a header row include first
        rows.append(headerow)
        trs = trs[1:]
    for tr in trs: # for every table row
        rows.append([td.get_text(strip=True) for td in tr.find_all('td')]) # data row
    return rows

def scrape_western(page, path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_guard = soup.find(id="voting-results-W_1")
    table_wrapper_forward = soup.find(id="voting-results-W_2")
    table_wrapper_center = soup.find(id="voting-results-W_3")
    table_guard = table_wrapper_guard.find('table')
    table_forward = table_wrapper_forward.find('table')
    table_center = table_wrapper_center.find('table')
    parsed_rows_guard = tableDataText(table_guard)
    parsed_rows_forward = tableDataText(table_forward)
    parsed_rows_center = tableDataText(table_center)
    write_file(path, 'table_guard_western', parsed_rows_guard[1:])
    write_file(path, 'table_forward_western', parsed_rows_forward[1:])
    write_file(path, 'table_center_western', parsed_rows_center[1:])

def scrape_eastern(page, path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_guard = soup.find(id="voting-results-E_1")
    table_wrapper_forward = soup.find(id="voting-results-E_2")
    table_wrapper_center = soup.find(id="voting-results-E_3")
    table_guard = table_wrapper_guard.find('table')
    table_forward = table_wrapper_forward.find('table')
    table_center = table_wrapper_center.find('table')
    parsed_rows_guard = tableDataText(table_guard)
    parsed_rows_forward = tableDataText(table_forward)
    parsed_rows_center = tableDataText(table_center)
    write_file(path, 'table_guard_eastern', parsed_rows_guard[1:])
    write_file(path, 'table_forward_eastern', parsed_rows_forward[1:])
    write_file(path, 'table_center_eastern', parsed_rows_center[1:])

def scrape_western_2013(page, path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_frontcourt = soup.find(id="voting-results-W_1")
    table_wrapper_backcourt = soup.find(id="voting-results-W_2")
    table_frontcourt = table_wrapper_frontcourt.find('table')
    table_backcourt = table_wrapper_backcourt.find('table')
    parsed_rows_frontcourt = tableDataText(table_frontcourt)
    parsed_rows_backcourt = tableDataText(table_backcourt)
    write_file(path, 'table_frontcourt_western', parsed_rows_frontcourt[1:])
    write_file(path, 'table_backcourt_western', parsed_rows_backcourt[1:])

def scrape_eastern_2013(page, path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_frontcourt = soup.find(id="voting-results-E_1")
    table_wrapper_backcourt = soup.find(id="voting-results-E_2")
    table_frontcourt = table_wrapper_frontcourt.find('table')
    table_backcourt = table_wrapper_backcourt.find('table')
    parsed_rows_frontcourt = tableDataText(table_frontcourt)
    parsed_rows_backcourt = tableDataText(table_backcourt)
    write_file(path, 'table_frontcourt_eastern', parsed_rows_frontcourt[1:])
    write_file(path, 'table_backcourt_eastern', parsed_rows_backcourt[1:])

def scrape_frontcourt_western(page, path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_frontcourt = soup.find(id="div_fc-w")
    table_frontcourt = table_wrapper_frontcourt.find('table')
    parsed_rows_frontcourt = tableDataText(table_frontcourt)
    write_file_2017(path, 'table_frontcourt_western', parsed_rows_frontcourt[2:])

def scrape_backcourt_western(page,path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_backcourt = soup.find(id="div_bc-w")
    table_backcourt = table_wrapper_backcourt.find('table')
    parsed_rows_backcourt = tableDataText(table_backcourt)
    write_file_2017(path, 'table_backcourt_western', parsed_rows_backcourt[2:])

def scrape_frontcourt_eastern(page, path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_frontcourt = soup.find(id="div_fc-e")
    table_frontcourt = table_wrapper_frontcourt.find('table')
    parsed_rows_frontcourt = tableDataText(table_frontcourt)
    write_file_2017(path, 'table_frontcourt_eastern', parsed_rows_frontcourt[2:])

def scrape_backcourt_eastern(page,path):
    soup = BeautifulSoup(page.content, 'html.parser');
    table_wrapper_backcourt = soup.find(id="div_bc-e")
    table_backcourt = table_wrapper_backcourt.find('table')
    parsed_rows_backcourt = tableDataText(table_backcourt)
    write_file_2017(path, 'table_backcourt_eastern', parsed_rows_backcourt[2:])

def prepare_page_and_make_folder(url, year):
    page = requests.get(url);
    path = r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year;
    if os.path.isdir(path) == False:
        os.mkdir(path)
    return page, path

def write_file(path, filename, rows):
    directory = os.path.realpath(path);
    foupath = os.path.join(directory, filename + ".csv");
    with open(foupath, 'w', encoding="utf-8", newline='') as file:
        w = csv.writer(file)
        w.writerow(['rank', 'name', 'votes'])
        for row in rows:
            temp = row[2].split(',')
            number = ""
            for temp_str in temp:
                number = number + temp_str
            w.writerow([row[0], row[1], int(number)])

def write_file_sorted(path, filename, rows):
    all_votes_sum = 0
    for row in rows:
        all_votes_sum = all_votes_sum + row[2]



    directory = os.path.realpath(path);
    foupath = os.path.join(directory, filename + ".csv");
    with open(foupath, 'w', encoding="utf-8", newline='') as file:
        w = csv.writer(file)
        w.writerow(['name', 'votes', 'all_star_share'])
        for row in rows:
            w.writerow([row[1], row[2], calculate_percentage(all_votes_sum, row[2])])


def write_file_2017(path, filename, rows):
    directory = os.path.realpath(path);
    foupath = os.path.join(directory, filename + ".csv");
    with open(foupath, 'w', encoding="utf-8", newline='') as file:
        w = csv.writer(file)
        w.writerow(['rank', 'name', 'votes'])
        for row in rows:
            temp = row[1].split(',')
            number = ""
            for temp_str in temp:
                number = number + temp_str
            w.writerow([row[2], row[0], int(number)])

def scrape_season_1980_2012():
    counter = 1979
    for index in range(1980, 2013):
        counter = counter + 1
        if counter == 1999:
            continue
        page, path = prepare_page_and_make_folder(
            "https://www.basketball-reference.com/allstar/NBA_" + str(index) + "_voting.html", str(index));
        scrape_western(page, path)
        scrape_eastern(page, path)

def scrape_season_2013_2016():
    for index in range(2013, 2017):
        page, path = prepare_page_and_make_folder("https://www.basketball-reference.com/allstar/NBA_" + str(index) + "_voting.html", str(index));
        scrape_western_2013(page, path)
        scrape_eastern_2013(page, path)

def scrape_season_2017_2020():
    for index in range(2017, 2021):
        page_eastern_conference_frontcourt, path = prepare_page_and_make_folder("https://www.basketball-reference.com/allstar/NBA_" + str(index) + "_voting-frontcourt-eastern-conference.html", str(index));
        page_eastern_conference_backcourt, path = prepare_page_and_make_folder("https://www.basketball-reference.com/allstar/NBA_" + str(index) + "_voting-backcourt-eastern-conference.html", str(index));
        page_western_conference_frontcourt, path = prepare_page_and_make_folder("https://www.basketball-reference.com/allstar/NBA_" + str(index) + "_voting-frontcourt-western-conference.html", str(index));
        page_western_conference_backcourt, path = prepare_page_and_make_folder("https://www.basketball-reference.com/allstar/NBA_" + str(index) + "_voting-backcourt-western-conference.html", str(index));

        scrape_backcourt_eastern(page_eastern_conference_backcourt, path)
        scrape_frontcourt_eastern(page_eastern_conference_frontcourt, path)
        scrape_frontcourt_western(page_western_conference_frontcourt, path)
        scrape_backcourt_western(page_western_conference_backcourt, path)

def sort_all_star_votes():
    counter = 1979
    for index in range(1980, 2013):
        counter = counter + 1
        if(counter == 1999):
            continue
        write_and_sort(str(index))

    for index in range(2013, 2020):
        write_and_sort_courts(str(index))


def read_csv(path, votes):
    with open(path, 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        counter = -1;
        for row in csv_reader:
            counter = counter + 1;
            if counter == 0:
                continue
            row[2] = int(row[2])
            votes.append(row)

def read_csv_sorted(path, votes, year):
    with open(path, 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        counter = -1;
        for row in csv_reader:
            counter = counter + 1
            if counter == 0:
                continue
            row[1] = int(row[1])
            season = year + 1;
            temp = str(season);
            votes.append([row[0], row[1], str(year) + "-" + str(temp[-2:]), row[2]])


def write_and_sort(year):
    votes = []
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_center_western.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-"+ year + r"\table_center_eastern.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year +  r"\table_guard_western.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_guard_eastern.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_forward_western.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_forward_eastern.csv",
        votes)
    votes_sorted = sorted(votes, key=lambda vote: vote[2], reverse=True)
    write_file_sorted(r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year, "sorted_votes_" + year, votes_sorted)

def write_and_sort_courts(year):
    votes = []
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_frontcourt_western.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_frontcourt_eastern.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_backcourt_western.csv",
        votes)
    read_csv(
        r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + r"\table_backcourt_eastern.csv",
        votes)
    print(votes)
    votes_sorted = sorted(votes, key=lambda vote: vote[2], reverse=True)
    print(votes_sorted)
    write_file_sorted(r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year,
                      "sorted_votes_" + year, votes_sorted)


def merge_files(year, votes):
    read_csv_sorted(r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year + "\sorted_votes_" + year + ".csv", votes, int(year))


def merge_all():
    votes = []
    counter = 1979
    for index in range(1980, 2020):
        counter = counter + 1
        if counter == 1999:
            continue
        merge_files(str(index), votes)
    with open("all_star_votes_history", 'w', encoding="utf-8", newline='') as file:
        w = csv.writer(file)
        w.writerow(['name', 'votes', 'season'])
        for row in votes:
            w.writerow(row)

def join_datasets(all_star_votes_history, mvp_votings):
    dataset = []
    all_star = all_star_votes_history[1::2]
    counter = -1
    for row_votings in mvp_votings:
        dataset.append([row_votings[0], row_votings[1], row_votings[2], row_votings[3], row_votings[4],
                                row_votings[5], row_votings[6], row_votings[7], row_votings[8], row_votings[9],
                                row_votings[10], row_votings[11],
                                row_votings[12], row_votings[13], row_votings[14], row_votings[15], row_votings[16],
                                row_votings[17], row_votings[18],
                                row_votings[19], row_votings[20], row_votings[21], row_votings[22], row_votings[23],
                                row_votings[24], row_votings[25], row_votings[26], 0, 0])

    for row in all_star_votes_history:
        counter = -1
        for row_votings in dataset:
            counter = counter + 1
            if row[0] == row_votings[9] and row[2] == row_votings[8]:
                dataset[counter] = [row_votings[0], row_votings[1], row_votings[2], row_votings[3], row_votings[4],
                                row_votings[5], row_votings[6], row_votings[7], row_votings[8], row_votings[9],
                                row_votings[10], row_votings[11],
                                row_votings[12], row_votings[13], row_votings[14], row_votings[15], row_votings[16],
                                row_votings[17], row_votings[18],
                                row_votings[19], row_votings[20], row_votings[21], row_votings[22], row_votings[23],
                                row_votings[24], row_votings[25], row_votings[26], row[1], row[3]];
                break

    return dataset

def join_datasets_test_data(all_star_votes_history, mvp_votings):
    dataset = []
    all_star = all_star_votes_history[1::2]
    counter = -1
    for row_votings in mvp_votings:
        dataset.append([row_votings[0], row_votings[1], row_votings[2], row_votings[3], row_votings[4],
                                row_votings[5], row_votings[6], row_votings[7], row_votings[8], row_votings[9],
                                row_votings[10], row_votings[11],
                                row_votings[12], row_votings[13], row_votings[14], row_votings[15], row_votings[16],
                                row_votings[17], row_votings[18],
                                row_votings[19], row_votings[20], row_votings[21], 0, 0])

    for row in all_star_votes_history:
        counter = -1
        for row_votings in dataset:
            counter = counter + 1
            if row[0] == row_votings[8]:
                dataset[counter] = [row_votings[0], row_votings[1], row_votings[2], row_votings[3], row_votings[4],
                                row_votings[5], row_votings[6], row_votings[7], row_votings[8], row_votings[9],
                                row_votings[10], row_votings[11],
                                row_votings[12], row_votings[13], row_votings[14], row_votings[15], row_votings[16],
                                row_votings[17], row_votings[18],
                                row_votings[19], row_votings[20], row_votings[21], row[1], row[2]];
                break

    return dataset


def write_joined_csv(dataset, path, filename):
        directory = os.path.realpath(path);
        foupath = os.path.join(directory, filename + ".csv");
        with open(foupath, 'w', encoding="utf-8", newline='') as file:
            w = csv.writer(file)
            # TODO dodaj header
            w.writerow(['', 'fga', 'fg3a', 'fta', 'per', 'ts_pct','usg_pct','bpm','season', 'player','win_pct','votes_first',
                        'points_won', 'points_max', 'award_share', 'g', 'mp_per_g', 'pts_per_g', 'trb_perg', 'ast_per_g',
                        'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws', 'ws_per_48','all_star_votes', 'all_star_share'])
            for row in dataset:
                w.writerow(row)


def write_joined_csv_test_data(dataset, path, filename):
    directory = os.path.realpath(path);
    foupath = os.path.join(directory, filename + ".csv");
    with open(foupath, 'w', encoding="utf-8", newline='') as file:
        w = csv.writer(file)
        # TODO dodaj header
        w.writerow(
            ['', 'fga', 'fg3a', 'fta', 'per', 'ts_pct', 'usg_pct', 'bpm', 'player', 'win_pct',
             'g', 'mp_per_g', 'pts_per_g', 'trb_perg', 'ast_per_g',
             'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws', 'ws_per_48', 'all_star_votes',
             'all_star_share'])
        for row in dataset:
            w.writerow(row)


def make_dataset(path):
    dataset = []
    with open(path, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        counter = -1;
        counter_2 = -1
        for row in reader:
            counter = counter + 1
            if (counter == 0):
                continue
            dataset.append(row)

    return dataset


def calculate_percentage(sum, votes):
    prct = votes/sum
    return round(prct,3)

def linear_regression():
    df = pd.read_csv("train.csv")
    dt = pd.read_csv("test.csv")
    reg = linear_model.LinearRegression()
    print('LINEAR START')
    reg.fit(df[['pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm']],df.award_share)
    rez = reg.predict(dt[['pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm']])
    
    print(rez)
    print(dt.award_share)
    print('LINEAR END')

def random_forest_regression(X, y):

    target = np.array(X['award_share'])
    features = X.drop('award_share', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features = features;
    train_labels = target;

    test_features = np.array(y.drop('award_share', axis=1))
    test_labels = np.array(y['award_share'])

    # print('Training Features Shape:', train_features.shape)
    # print('Training Labels Shape:', train_labels.shape)
    # print('Testing Features Shape:', test_features.shape)
    # print('Testing Labels Shape:', test_labels.shape)

    r1 = RandomForestRegressor(n_estimators=1000, random_state=0, bootstrap=True)
    r2 = RandomForestRegressor(n_estimators=500, random_state=0, bootstrap=True)
    r1.fit(train_features, train_labels)
    r2.fit(train_features, train_labels)

    predictions_r1 = r1.predict(test_features)
    predictions_r2 = r2.predict(test_features)

    data = pd.read_csv('test.csv', usecols=['player', 'season', 'award_share'])
    players = np.array(data['player'])
    seasons = np.array(data['season'])
    award_shares = np.array(data['award_share'])
    tuples = list(zip(players, seasons, award_shares, predictions_r1))
    df = pd.DataFrame(tuples, columns=['Player', 'Season', 'Award_share', 'Predictions'])

    errors_r1 = abs(predictions_r1 - test_labels)
    errors_r2 = abs(predictions_r2 - test_labels)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error r1:', round(np.mean(errors_r1), 2), '.')
    print('Mean Absolute Error r2:', round(np.mean(errors_r2), 2), '.')

    print("Score r1: " + str(r1.score(test_features, test_labels)))
    print("Score r2: " + str(r2.score(test_features, test_labels)))

    return df

if __name__ == '__main__':

    # scrape_season_1980_2012()
    # scrape_season_2013_2016()
    # scrape_season_2017_2020()
    #
    # sort_all_star_votes()
    # merge_all()
    #
    # all_star_history_votes = make_dataset(r'C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all_star_votes_history')
    # mvp_votings = make_dataset(r'C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\mvp_votings.csv')
    # dataset = join_datasets(all_star_history_votes, mvp_votings)
    # write_joined_csv(dataset, r'C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping', 'mvp_votings_joined')
    #
    # # pravljenje novog test_data sa all_star_share
    # all_star_votes_2019 = make_dataset(r'C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-2019\sorted_votes_2019.csv')
    # test_data = make_dataset(r'C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\test_data.csv')
    # dataset_test_data = join_datasets_test_data(all_star_votes_2019, test_data)
    # write_joined_csv_test_data(dataset_test_data, r'C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping', 'test_data_joined')

    # PEARSON CORRELATION
    X = pd.read_csv('mvp_votings_joined.csv')
    corr = X.corr()
    coef = corr['award_share'].sort_values(ascending=False)
    coef.to_csv('person_correlations', sep='\t', encoding='utf-8')
    # RANDOM FOREST
    X_all_star = pd.read_csv('train.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm', 'all_star_share'])
    y_all_star = pd.read_csv('test.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm', 'all_star_share'])
    df_all_star = random_forest_regression(X_all_star, y_all_star)
    df_all_star.to_csv('predicted_values_all_star', sep='\t', encoding='utf-8')
    #LINEAR
    linear_regression()


    X = pd.read_csv('train.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm'])
    y = pd.read_csv('test.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm'])
    df = random_forest_regression(X, y)
    df.to_csv('predicted_values', sep='\t', encoding='utf-8')
