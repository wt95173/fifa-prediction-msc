import pandas as pd
from joblib import load

rf_model = load('../models/sgd_model.joblib')
df_base = pd.read_csv('../dataset/results2/df_base_new.csv')
df_team_stats = pd.read_csv('../dataset/results/df_team_stats.csv')

wc_group = {'A': [['Qatar', 0], ['Ecuador', 0], ['Senegal', 0], ['Netherlands', 0]],
            'B': [['England', 0], ['Iran', 0], ['United States', 0], ['Wales', 0]],
            'C': [['Argentina', 0], ['Saudi Arabia', 0], ['Mexico', 0], ['Poland', 0]],
            'D': [['France', 0], ['Australia', 0], ['Denmark', 0], ['Tunisia', 0]],
            'E': [['Spain', 0], ['Costa Rica', 0], ['Germany', 0], ['Japan', 0]],
            'F': [['Belgium', 0], ['Canada', 0], ['Morocco', 0], ['Croatia', 0]],
            'G': [['Brazil', 0], ['Serbia', 0], ['Switzerland', 0], ['Cameroon', 0]],
            'H': [['Portugal', 0], ['Ghana', 0], ['Uruguay', 0], ['South Korea', 0]]}


def get_data(team_name):
    past_games = df_team_stats.query('team == @team_name').sort_values(by='date', ascending=False)
    last5 = past_games.head(5)
    stats_dict = {}

    stats_dict["rank_last"] = past_games["rank"].values[0]
    stats_dict["goals"] = past_games["score"].mean()
    stats_dict["goals_l5"] = last5["score"].mean()
    stats_dict["goals_ano"] = past_games["score_ano"].mean()
    stats_dict["goals_ano_l5"] = last5["score_ano"].mean()
    stats_dict["rank"] = past_games["rank_ano"].mean()
    stats_dict["rank_l5"] = last5["rank_ano"].mean()
    if len(last5) > 0:
        stats_dict["points"] = past_games["total_points"].values[0] - past_games["total_points"].values[-1]
        stats_dict["points_l5"] = last5["total_points"].values[0] - last5["total_points"].values[-1]
    else:
        stats_dict["points"] = 0
        stats_dict["points_l5"] = 0
    stats_dict["gp"] = past_games["team_points"].mean()
    stats_dict["gp_l5"] = last5["team_points"].mean()
    stats_dict["gp_rank"] = past_games["points_by_rank"].mean()
    stats_dict["gp_rank_l5"] = last5["points_by_rank"].mean()
    stats_dict["pm"] = past_games["player_mean_dif"].mean()
    stats_dict["pm_l5"] = last5["player_mean_dif"].mean()

    return stats_dict


def get_features(home_team, away_team):
    rank_dif = home_team["rank_last"] - away_team["rank_last"]
    goals_dif = home_team["goals"] - away_team["goals"]
    goals_l5_dif = home_team["goals_l5"] - away_team["goals_l5"]
    goals_ano_dif = home_team["goals_ano"] - away_team["goals_ano"]
    goals_l5_ano_dif = home_team["goals_ano_l5"] - away_team["goals_ano_l5"]
    rank_mean_dif = home_team["rank"] - away_team["rank"]
    rank_mean_l5_dif = home_team["rank_l5"] - away_team["rank_l5"]
    points_mean_dif = home_team["points"] - away_team["points"]
    points_mean_l5_dif = home_team["points_l5"] - away_team["points_l5"]
    game_points_dif = home_team["gp"] - away_team["gp"]
    game_points_l5_dif = home_team["gp_l5"] - away_team["gp_l5"]
    game_points_rank_dif = home_team["gp_rank"] - away_team["gp_rank"]
    game_points_rank_l5_dif = home_team["gp_rank_l5"] - away_team["gp_rank_l5"]
    player_dif_mean_dif = home_team["pm"] - away_team["pm"]
    player_dif_mean_l5_dif = home_team["pm_l5"] - away_team["pm_l5"]

    feature = [rank_dif, goals_dif, goals_l5_dif, goals_ano_dif, goals_l5_ano_dif,
               rank_mean_dif, rank_mean_l5_dif, points_mean_dif, points_mean_l5_dif, game_points_dif,
               game_points_l5_dif, game_points_rank_dif, game_points_rank_l5_dif, player_dif_mean_dif,
               player_dif_mean_l5_dif, 1]

    return feature


def simulation_match(team1, team2):
    home_data = get_data(team1)
    away_data = get_data(team2)

    features1 = get_features(home_data, away_data)
    features2 = get_features(away_data, home_data)

    prob1 = rf_model.predict_proba([features1])
    prob2 = rf_model.predict_proba([features2])

    print(team1, team2)

    if (prob1[0][0] > prob1[0][1] and prob2[0][0] < prob2[0][1]) or (
            prob1[0][0] < prob1[0][1] and prob2[0][0] > prob2[0][1]):
        print(prob1, prob2, 'Draw')
        return 1

    if prob1[0][0] > prob1[0][1] and prob2[0][0] > prob2[0][1]:
        print(prob1, prob2, 'Win')
        return 2

    if prob1[0][0] < prob1[0][1] and prob2[0][0] < prob2[0][1]:
        print(prob1, prob2, 'Lose')
        return 0


def update_score(wc_group_data, group, team1_idx, team2_idx):
    result = simulation_match(wc_group_data[group][team1_idx][0], wc_group_data[group][team2_idx][0])

    if result == 2:
        wc_group_data[group][team1_idx][1] += 3
    elif result == 0:
        wc_group_data[group][team2_idx][1] += 3
    elif result == 1:
        wc_group_data[group][team1_idx][1] += 1
        wc_group_data[group][team2_idx][1] += 1


def generate_matches_in_group(group):
    n = len(group)
    matches = []
    for i in range(n):
        for j in range(i + 1, n):
            matches.append((i, j))
    return matches


def simulate(wc_group_data):
    for group, teams in wc_group_data.items():
        matches = generate_matches_in_group(teams)
        print(group)
        for team1_idx, team2_idx in matches:
            update_score(wc_group_data, group, team1_idx, team2_idx)
    return wc_group_data


for item in simulate(wc_group).items():
    print(item)

team_16 = {'A': ['Senegal', 'Netherlands'],
           'B': ['England', 'Iran'],
           'C': ['Argentina', 'Mexico'],
           'D': ['France', 'Australia'],
           'E': ['Spain', 'Japan'],
           'F': ['Belgium', 'Croatia'],
           'G': ['Brazil', 'Cameroon'],
           'H': ['Portugal', 'Uruguay']}


def simulation_match2(team1, team2):
    home_data = get_data(team1)
    away_data = get_data(team2)

    features1 = get_features(home_data, away_data)
    features2 = get_features(away_data, home_data)

    prob1 = rf_model.predict_proba([features1])
    prob2 = rf_model.predict_proba([features2])

    team1_prob = (prob1[0][0] + prob2[0][1]) / 2
    team2_prob = (prob1[0][1] + prob2[0][0]) / 2

    print(team1, team2, team1_prob, team2_prob)

    if team1_prob > team2_prob:
        return 1
    else:
        return 0


def next_round(matches):
    next_round_matches = []
    x1 = ''
    x2 = ''

    if len(matches) == 1:
        if simulation_match2(matches[0][0], matches[0][1]) == 1:
            return matches[0][0]
        else:
            return matches[0][1]
    else:
        for match in matches:
            if simulation_match2(match[0], match[1]) == 1:
                if x1 == '':
                    x1 = match[0]
                else:
                    x2 = match[0]
            else:
                if x1 == '':
                    x1 = match[1]
                else:
                    x2 = match[1]

            if x1 != '' and x2 != '':
                next_round_matches.append((x1, x2))
                x1 = ''
                x2 = ''

        return next_round_matches


def simulation_final(team_16_data):
    matches_16 = [(team_16_data['A'][0], team_16_data['B'][1]),
                  (team_16_data['C'][0], team_16_data['D'][1]),
                  (team_16_data['E'][0], team_16_data['F'][1]),
                  (team_16_data['G'][0], team_16_data['H'][1]),
                  (team_16_data['B'][0], team_16_data['A'][1]),
                  (team_16_data['D'][0], team_16_data['C'][1]),
                  (team_16_data['F'][0], team_16_data['E'][1]),
                  (team_16_data['H'][0], team_16_data['G'][1])]
    print('16:')
    matches_8 = next_round(matches_16)
    print('8:')
    matches_4 = next_round(matches_8)
    print('semi-finals')
    matches_2 = next_round(matches_4)
    print('finals')
    champions = next_round(matches_2)
    return champions


print(simulation_final(team_16))
