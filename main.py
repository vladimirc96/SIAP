import pydot as pydot
import requests
from bs4 import BeautifulSoup
import os
import csv
import numpy as np
from chefboost import Chefboost as chef
from six import StringIO
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import importlib.util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.model_selection import GridSearchCV
# Scatterplot Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz

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
    return all_votes_sum

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
    votes_per_season = []
    for index in range(1980, 2013):
        counter = counter + 1
        if(counter == 1999):
            continue
        votes_per_season.append([index, write_and_sort(str(index))])

    for index in range(2013, 2020):
        votes_per_season.append([index, write_and_sort_courts(str(index))])

    return votes_per_season


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
    sum = write_file_sorted(r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year, "sorted_votes_" + year, votes_sorted)
    return sum

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
    sum = write_file_sorted(r"C:\Users\Vladimir\PycharmProjects\all-star-votings-scrapping\all-star-votings-" + year,
                      "sorted_votes_" + year, votes_sorted)
    return sum


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


def gradient_boosting(year):
    df = pd.read_csv("train_gb.csv")
    dt = pd.read_csv("test_"+str(year)+"_gb"+".csv")
    num_of_instances = dt.shape[0]
    # test_labels = np.array(dt['Decision'])
    config = {'enableGBM': True, 'epochs':7,'learning_rate': 1}
    df.head()
    model = chef.fit(df[['pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm', 'Decision']], config)
    mae = 0
    predictions = []
    for index, instance in dt.iterrows():
        actual = instance['Decision']
        prediction = 0
        for j in  range(0,7):
            moduleName  = "outputs/rules/rules%s" % (j)
            # fp, pathName, description = imp.find_module(moduleName)
            # myRules = imp.load_module(moduleName,fp,pathName,description)
            tree = chef.restoreTree(moduleName)

            prediction = prediction+ tree.findDecision( instance.values)

        print("Actual: ",actual,"Prediction: ", prediction)
        error = abs(actual - prediction)
        mae = mae + error
        predictions.append(prediction)
    mae = mae/num_of_instances
    print("Mean Absolute Error GB: ", mae)


    data = pd.read_csv('test_' + str(year) + '.csv', usecols=['player', 'season', 'award_share', 'per', 'ws', 'ws_per_48', 'bpm'])
    players = np.array(data['player'])
    seasons = np.array(data['season'])
    award_shares = np.array(data['award_share'])
    per = np.array(data['per'])
    ws = np.array(data['ws'])
    ws_per_48 = np.array(data['ws_per_48'])
    bpm = np.array(data['bpm'])
    tuples = list(zip(players, seasons, award_shares, predictions, per, ws, ws_per_48, bpm))
    dataFrame = pd.DataFrame(tuples, columns=['player', 'season', 'award_share', 'predictions', 'per', 'ws', 'ws_per_48', 'bpm'])

    return dataFrame

def linear_regression():
    df = pd.read_csv("train.csv")
    dt = pd.read_csv("test.csv")
    test_labels = np.array(dt['award_share'])
    reg = linear_model.LinearRegression()
    print('LINEAR START')
    reg.fit(df[['pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm']],df.award_share)

    prediction = reg.predict(dt[['pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm']])

    print('Mean Absolute Error L:', round(mean_absolute_error(test_labels, prediction), 2), '.')


    print('LINEAR END')

def random_forest_regression(X, y, year):

    target = np.array(X['award_share'])
    features = X.drop('award_share', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features = features;
    train_labels = target;

    test_features = np.array(y.drop('award_share', axis=1))
    test_labels = np.array(y['award_share'])

    r1 = RandomForestRegressor(n_estimators=1500, random_state=42, bootstrap=True, max_depth=80,
                               max_features='sqrt', min_samples_leaf=5, min_samples_split=3)
    r1.fit(train_features, train_labels)
    predictions_r1 = r1.predict(test_features)

    evaluate(r1, test_features, test_labels)

    data = pd.read_csv('test_' + str(year) + '.csv', usecols=['player', 'season', 'award_share',  'pts_per_g','per', 'ws', 'bpm', 'all_star_share'])
    players = np.array(data['player'])
    seasons = np.array(data['season'])
    award_shares = np.array(data['award_share'])
    per = np.array(data['per'])
    ws = np.array(data['ws'])
    pts_per_g = np.array(data['pts_per_g'])
    bpm = np.array(data['bpm'])
    all_star_share = np.array(data['all_star_share'])
    tuples = list(zip(players, seasons, award_shares, predictions_r1, per, ws, bpm, pts_per_g, all_star_share))
    df = pd.DataFrame(tuples, columns=['player', 'season', 'award_share', 'predictions', 'per', 'ws', 'bpm', 'pts_per_g', 'all_star_share'])

    return df


def make_random_forest_regressor(is_all_star, X):
    target = np.array(X['award_share'])
    features = X.drop('award_share', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features = features;
    train_labels = target;

    r = RandomForestRegressor()
    if is_all_star == True:
        r = RandomForestRegressor(n_estimators=1000, random_state=42, bootstrap=True, max_depth=40,
                                   max_features='sqrt', min_samples_leaf=5, min_samples_split=10)
        r.fit(train_features, train_labels)
    else:
        r = RandomForestRegressor(n_estimators=1500, random_state=42, bootstrap=True, max_depth=80,
                                   max_features='sqrt', min_samples_leaf=5, min_samples_split=3)
        r.fit(train_features, train_labels)

    return r;

def predict(regressor, X, y, year):

    target = np.array(X['award_share'])
    features = X.drop('award_share', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features = features;
    train_labels = target;

    test_features = np.array(y.drop('award_share', axis=1))
    test_labels = np.array(y['award_share'])
    predictions_r1 = regressor.predict(test_features)

    evaluate(regressor, test_features, test_labels)

    data = pd.read_csv('test_' + str(year) + '.csv', usecols=['player', 'season', 'award_share', 'pts_per_g','per', 'ws', 'bpm', 'all_star_share'])
    players = np.array(data['player'])
    seasons = np.array(data['season'])
    award_shares = np.array(data['award_share'])
    per = np.array(data['per'])
    ws = np.array(data['ws'])
    pts_per_g = np.array(data['pts_per_g'])
    bpm = np.array(data['bpm'])
    all_star_share = np.array(data['all_star_share'])
    tuples = list(zip(players, seasons, award_shares, predictions_r1, per, ws, bpm, pts_per_g, all_star_share))
    df = pd.DataFrame(tuples, columns=['player', 'season', 'award_share', 'predictions', 'per', 'ws', 'bpm', 'pts_per_g', 'all_star_share'])

    return df


def check_results(dataframe):
    real = dataframe.sort_values('award_share', ascending=False)
    prediction = dataframe.sort_values('predictions', ascending=False)
    # fig, ax = plt.subplots(1,1)
    # ax.axis('off')
    # pd.plotting.table(ax,real)
    # pd.plotting.table(ax,prediction)
    # plt.show()
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    print('\n')
    print(real[0:6])
    print('\n')
    print(prediction[0:6])
    print('\n')
    return real[0:6], prediction[0:6]


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    print('Model Performance')
    print('Mean absolute error: {:0.4f} degrees.'.format(mean_absolute_error(test_labels, predictions)))
    print('R2 score: ' + str(r2_score(test_labels, predictions)))

def random_search(X,y):
    # get train and test features and labels
    target = np.array(X['award_share'])
    features = X.drop('award_share', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features = features;
    train_labels = target;
    test_features = np.array(y.drop('award_share', axis=1))
    test_labels = np.array(y['award_share'])

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)

    # RANDOM SEARCH
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(train_features, train_labels)
    rf_random.best_params_
    print(rf_random.best_params_)
    pprint(rf_random.best_params_)

    # Compare randomized and original
    base_model = RandomForestRegressor(n_estimators=1000, random_state=42, bootstrap=True)
    base_model.fit(train_features, train_labels)
    base_model_predictions = base_model.predict(test_features)

    best_random = rf_random.best_estimator_
    best_random_predictions = best_random.predict(test_features)

    errors_base_model = abs(base_model_predictions - test_labels)
    errors_best_random_model = abs(best_random_predictions - test_labels)

    print("************** Evaluacija za best_random i base_model **************")
    evaluate(best_random, test_features, test_labels)
    evaluate(base_model, test_features, test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error base:', round(mean_absolute_error(test_labels, base_model_predictions), 5), '.')
    print('Mean Absolute Error random:', round(mean_absolute_error(test_labels, best_random_predictions), 5), '.')

def grid_search(X, y):
    # get train and test features and labels
    target = np.array(X['award_share'])
    features = X.drop('award_share', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features = features;
    train_labels = target;
    test_features = np.array(y.drop('award_share', axis=1))
    test_labels = np.array(y['award_share'])

    # GRID SEARCH
    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': ['sqrt'],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'n_estimators': [1500, 1800, 2000, 2200]
    }

    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                              cv = 5, n_jobs = -1, verbose = 2)
    grid_search.fit(train_features, train_labels)

    print("************** Evaluacija za grid_search **************")
    grid_search.best_params_
    pprint(grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    evaluate(best_grid, test_features, test_labels)


if __name__ == '__main__':
    # scrape_season_1980_2012()
    # scrape_season_2013_2016()
    # scrape_season_2017_2020()
    # votes_per_season = []
    # # ovde mogu dobiti za svaku god ukupan broj glasova
    # votes_per_season = sort_all_star_votes()
    # data_frame_votes_per_season = pd.DataFrame(votes_per_season,columns=['season', 'votes'])
    # data_frame_votes_per_season.to_csv('votes_per_season.csv', sep='\t', encoding='utf-8')
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
    #

    # PEARSON CORRELATION
    X = pd.read_csv('mvp_votings_joined.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'ws_per_48', 'bpm', 'all_star_share'])
    # X = pd.read_csv('mvp_votings_joined.csv')
    corr = X.corr()
    coef = corr['award_share'].sort_values(ascending=False)
    corr.to_csv('person_correlations.csv', sep='\t', encoding='utf-8')

    # plt.scatter(X['per'], X['award_share'])
    # plt.xlabel('per')
    # plt.ylabel('award_share')
    # plt.show()
    # plt.scatter(X['ws'], X['award_share'])
    # plt.xlabel('ws')
    # plt.ylabel('award_share')
    # plt.show()
    # TREND GLASOVA KROZ GODINE
    # votes_per_season = pd.read_csv('votes_per_season.csv', delimiter='\t')
    # plt.plot(votes_per_season['season'], votes_per_season['votes'])
    # plt.xlabel('season')
    # plt.ylabel('votes')
    # plt.show()
    # plt.scatter(X['all_star_share'], X['award_share'])
    # plt.xlabel('all_star_share')
    # plt.ylabel('award_share')
    # plt.show()


    # RANDOM FOREST
    X = pd.read_csv('train.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'bpm'])
    y = pd.read_csv('test_2013.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'bpm'])
    X_all_star = pd.read_csv('train.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'bpm', 'all_star_share'])
    y_all_star = pd.read_csv('test_2013.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'bpm', 'all_star_share'])
    # random_search(X,y)
    # random_search(X_all_star,y_all_star)
    # grid_search(X, y)
    # grid_search(X_all_star, y_all_star)
    regressor_all_star = make_random_forest_regressor(True, X_all_star)
    regressor = make_random_forest_regressor(False, X)

    for index in range(2013,2018):
        print('****************** ALL_STAR_SHARE ******************  \n')
        print("Season:" + str(index))
        y_all_star = pd.read_csv('test_' + str(index) + '.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'bpm', 'all_star_share'])
        df_all_star = predict(regressor_all_star,X_all_star, y_all_star, index)
        real_all_star, predicition_all_star = check_results(df_all_star)
        print('\n')
        print('****************** ALL_STAR_SHARE ******************  \n\n\n')

        print('****************** NO_ALL_STAR_SHARE ******************  ')
        print("Season:" + str(index))
        y = pd.read_csv('test_' + str(index) +  '.csv', usecols=['award_share', 'pts_per_g', 'per', 'ws', 'bpm'])
        df = predict(regressor, X, y, index)
        real, prediction = check_results(df)
        print('\n')
        print('****************** NO_ALL_STAR_SHARE ******************  \n')

        with open('predictions_all_star_' + str(index) + '.csv', 'w') as f:
            real_all_star.to_csv(f)
        with open('predictions_all_star_' + str(index) + '.csv', 'a') as f:
            predicition_all_star.to_csv(f)

        with open('predictions_' + str(index) + '.csv', 'w') as f:
            real.to_csv(f)
        with open('predictions_' + str(index) + '.csv', 'a') as f:
            prediction.to_csv(f)

    features = X.drop('award_share', axis=1)
    feature_list = list(features.columns)
    importances = regressor.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_list[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    features_all_star = X_all_star.drop('award_share', axis=1)
    feature_list_all_star = list(features_all_star.columns)
    importances_all_star = regressor_all_star.feature_importances_
    indices_all_star = np.argsort(importances_all_star)
    plt.title('Feature Importances All star')
    plt.barh(range(len(indices_all_star)), importances_all_star[indices_all_star], color='b', align='center')
    plt.yticks(range(len(indices_all_star)), [feature_list_all_star[i] for i in indices_all_star])
    plt.xlabel('Relative Importance')
    plt.show()

    tree = regressor.estimators_
    estimator_one = regressor.estimators_[0]
    estimator_full = regressor.estimators_
    # Export as dot file
    export_graphviz(estimator_one, out_file='tree_one.dot',
                    feature_names=features.columns,
                    rounded=True, proportion=False,
                    precision=2, filled=True)
    export_graphviz(estimator_full, out_file='tree_full.dot',
                    feature_names=features.columns,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    # #LINEAR
    # linear_regression()

    # # DECISION_TREE
    # for index in range(2013, 2018):
    #     df = gradient_boosting(index)
    #     real, prediction = check_results(df)
    #     pd.concat([real, prediction], axis=1).to_csv('predictions_' + str(index)+"_gb" + '.csv')

