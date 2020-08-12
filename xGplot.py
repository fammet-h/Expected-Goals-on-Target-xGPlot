import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pp
import category_encoders as ce
from PIL import Image
import football_pitch as pitch

im = Image.open("TH433.png").convert('P')

select_match_id = 16248

def freeze_frame_analysis(location, freeze_frame_list):
    """
    INPUT
        location: list of a shot location
        freeze_frame_list: list of freeze_frame of a shot
    OUTPUT
        num_def_players: (int) num of defenders around the area berween a shot-taker and the goal
        num_att_players: (int) num of attackers around the area berween a shot-taker and the goal
        GK_distance: distance between the center of the goal and a GK
        GK_angle: angle between the line of a shot-taker to the center of the goal 
					and the line of a GK to the goal center
    """
    num_def_players = 0
    num_att_players = 0
    GK_distance = None
    GK_angle = None
    for p in freeze_frame_list:
        if p['position']['name'] == "Goalkeeper":
            GK_distance = np.sqrt((p['location'][0] - 120) ** 2 + (p['location'][1] - 40) ** 2)
            # inner product of vector GK & vector shoter, at origin (120, 40)
            vec_GK = np.array([120 - p['location'][0], p['location'][1] - 40])
            vec_S = np.array([120 - location[0], location[1] - 40])
            cosine = np.inner(vec_GK, vec_S) / (np.linalg.norm(vec_GK) * np.linalg.norm(vec_S))
            GK_angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        if p['location'][0] < location[0] - 1:
            continue
        upper_bound = max(45, location[1] - 39)
        lower_bound = min(35, location[1] - 41)
        if p['location'][1] < lower_bound or upper_bound < p['location'][1]:
            continue
        if p['teammate']:
            num_att_players += 1
        else:
            num_def_players += 1
    return num_def_players, num_att_players, GK_distance, GK_angle

def train_calc_split(pd_shots, match_id, features, label='is_goal'):
    """
    INPUT
        pd_shots: (pandas) shots data (all type / on Target)
        match_id: statsbomb match_id
        features: list of features (column names)
        label: label column name
    OUTPUT
        train_x: shots data
        calc_x: shots data of the specified match_id
        train_y: label data
        calc_y: label data of the specified match_id
    """
    pd_train = pd_shots[pd_shots['match_id'] != match_id]
    train_x = pd_train[features]
    train_y = pd_train[label]
    pd_calc = pd_shots[pd_shots['match_id'] == match_id]
    calc_x = pd_calc[features]
    calc_y = pd_calc[label]
    return train_x, calc_x, train_y, calc_y

# statsbomb shot data
base_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
comp_url = base_url + "matches/{}/{}.json"
match_url = base_url + "events/{}.json"
competitions_url = base_url + "competitions.json"

competitions = requests.get(url=competitions_url).json()
comp_season_ids = [
	[cs['competition_id'], cs['season_id']] for cs in competitions if cs['competition_name'] == "La Liga"
]
matches = []
for cs in comp_season_ids:
    matches += requests.get(url=comp_url.format(cs[0], cs[1])).json()

all_shots = []
for match in tqdm(matches):
    match_id = match['match_id']
    events = requests.get(url=match_url.format(match_id)).json()
    shots = [x for x in events if x['type']['name'] == "Shot"]

    for s in shots:
        s_shot = s['shot']
        # calculate angle
        vec_L = np.array([120 - s['location'][0], s['location'][1] - 38])
        vec_R = np.array([120 - s['location'][0], s['location'][1] - 42])
        cosine = np.inner(vec_L, vec_R) / (np.linalg.norm(vec_L) * np.linalg.norm(vec_R))
        angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        # calculate num_def_players, num_att_players, GK_distance, GK_angle
        if 'freeze_frame' in s_shot:
            num_def_players, num_att_players, GK_distance, GK_angle = freeze_frame_analysis(s['location'], s_shot['freeze_frame'])
        else:
            num_def_players = None
            num_att_players = None
            GK_distance = None
            GK_angle = None
        
        # append each shot
        attributes = {
            "id": s['id'],
            'match_id': match_id,
            'match_date': match['match_date'],
            'home': match['home_team']['home_team_name'],
            'away': match['away_team']['away_team_name'],
            'home_score': match['home_score'],
            'away_score': match['away_score'],
            "period": s['period'],
            "minute": s['minute'],
            "second": s['second'],
            "possession_chain": s['possession'],
            "team": s['possession_team']['name'],
            "player": s['player']['name'], 
            "position": s['position']['name'],
            'x': s['location'][0],
            'y': s['location'][1],
            "distance": np.sqrt((120 - s['location'][0]) ** 2 + (40 - s['location'][1]) ** 2),
            "angle": angle,
            "duration": s['duration'],
            "related_events": s['related_events'],
            "body_part": s_shot['body_part']['name'],
            "play_type": s_shot['type']['name'],
            "is_goal": 1 if s_shot['outcome']['name'] == "Goal" else 0, ##### label #####
            "is_on_T": 0 if s_shot['outcome']['name'] in ['Blocked', 'Off T', 'Wayward', 'Saved Off T'] else 1,
            # Goal, Post, Saved, Saved To Post
            "sb_xg": s_shot['statsbomb_xg'],
            "height": s_shot['end_location'][2] if len(s_shot['end_location']) >= 3 else None,
            "side": 40 - s_shot['end_location'][1],
            "defenders": num_def_players,
            "attackers": num_att_players,
            'GK_distance': GK_distance, 
            'GK_angle': GK_angle,
            "is_under_pressure": 1 if 'under_pressure' in s and s['under_pressure'] else 0,
        }
        all_shots.append(attributes)

pd_shots = pd.DataFrame(all_shots)
pd_shots = pd_shots.set_index('id')

# OneHotEncoding 'body_part', 'play_type'
ce_ohe = ce.OneHotEncoder(cols=['body_part', 'play_type'])
pd_shots = ce_ohe.fit_transform(pd_shots)

pre_shot_features = [
    "distance", 
    "angle", 
    "duration", 
    'body_part_1', 'body_part_2', 'body_part_3', 'body_part_4', 
    'play_type_1', 'play_type_2', 'play_type_3', 
    "defenders", 
    "attackers", 
    'GK_distance', 
    'GK_angle', 
    "is_under_pressure"
]
post_shot_features = [
    "distance", 
    "angle", 
    'body_part_1', 'body_part_2', 'body_part_3', 'body_part_4', 
    "height", 
    "side", 
    "defenders", 
    "attackers", 
    'GK_distance', 
    'GK_angle'
]

pd_shots_onT = pd_shots[pd_shots.is_on_T == 1]
pd_is_goal = pd_shots_onT.is_goal

train_x, calc_x, train_y, calc_y = train_calc_split(pd_shots_onT, match_id=select_match_id, features=post_shot_features, label='is_goal')
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=True)

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

evals = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}
bst = xgb.train(
    {'objective': 'binary:logistic', 'eval_metric': 'logloss'}, 
    dtrain, 
    num_boost_round=100, 
    evals=evals, 
    evals_result=evals_result, 
    early_stopping_rounds=10
)

pred_proba = bst.predict(dtest)
pred = np.where(pred_proba > 0.5, 1, 0)
acc = accuracy_score(test_y, pred)

train_metric = evals_result['train']['logloss']
plt.plot(train_metric, label='train logloss')
eval_metric = evals_result['eval']['logloss']
plt.plot(eval_metric, label='eval logloss')
plt.grid()
plt.legend()
plt.xlabel('rounds')
plt.ylabel('logloss')
plt.show()

# xgb.plot_importance(bst)

dcalc = xgb.DMatrix(calc_x, label=calc_y)
pred = bst.predict(dcalc)
pd_calc = calc_x
pd_calc['is_goal'] = calc_y
pd_calc['xGonT'] = pred
pd_match_shots = pd.merge(
	pd_shots[pd_shots['match_id'] == select_match_id], 
	pd_calc[['xGonT']], 
	on='id', how='outer'
)
pd_match_shots = pd_match_shots.sort_values(['period', 'minute', 'second'])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])
pitch.soccer_pitch_h(ax)

for row in pd_match_shots.itertuples():
    pitch.xGonT_circle(
		ax, row.x, row.y, row.sb_xg, row.xGonT, goal=(row.is_goal == 1), home=(row.home == row.team)
	)

home_xG = [0 for i in range(90)]
away_xG = [0 for i in range(90)]
home_xGonT = [0 for i in range(90)]
away_xGonT = [0 for i in range(90)]

for row in pd_match_shots.itertuples():
    if row.period > 2:
        continue
    if row.period == 1 and row.minute >= 45:
        time = 44
    elif row.minute >= 90:
        time = 89
    else:
        time = row.minute
    
    if row.home == row.team:
        home_xG[time] += row.sb_xg
        if row.is_on_T == 1:
            home_xGonT[time] += row.xGonT
    else:
        away_xG[time] += row.sb_xg
        if row.is_on_T == 1:
            away_xGonT[time] += row.xGonT

plt.text(0, 35, 'xGplot', fontsize=16, color='blue', ha='center', va='bottom')
plt.text(0, 30, row.home + ' v ' + row.away, fontsize=16, color='black', ha='center', va='bottom')
plt.text(0, 25, row.match_date, fontsize=16, color='black', ha='center', va='bottom')
plt.text(0, 20, str(round(sum(home_xG), 3)) + ' xG ' + str(round(sum(away_xG), 3)), fontsize=16, color='black', ha='center', va='bottom')
plt.text(0, 15, str(round(sum(home_xGonT), 3)) + ' xGonT ' + str(round(sum(away_xGonT), 3)), fontsize=16, color='black', ha='center', va='bottom')
plt.text(0, 10, str(row.home_score) + ' Score ' + str(row.away_score), fontsize=16, color='black', ha='center', va='bottom')

plt.text(0, -35, 'offT: non fill; onT: blue to yellow as xGonT gets high; goal: red circle', fontsize=10, color='black', ha='center', va='bottom')

extent = [-30, 30, -30, 30]
ax.imshow(im, alpha=0.1, extent=extent)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.show()

for i in range(89):
    home_xG[i + 1] += home_xG[i]
    away_xG[i + 1] += away_xG[i]
    home_xGonT[i + 1] += home_xGonT[i]
    away_xGonT[i + 1] += away_xGonT[i]

match_time = list(range(90))

fig, ax = plt.subplots()

ax.plot(match_time, home_xG, color=(0.0, 0.0, 1.0), linewidth=2, label='home_xG')
ax.plot(match_time, away_xG, color=(1.0, 0.0, 0.0), linewidth=2, label='away_xG')
ax.plot(match_time, home_xGonT, color=(0.0, 0.0, 0.7), linewidth=2, label='home_xGonT')
ax.plot(match_time, away_xGonT, color=(0.7, 0.0, 0.0), linewidth=2, label='away_xGonT')

ax.set_xlabel('time')
ax.set_ylabel('xG')
ax.set_title('xGplot')
ax.margins(x=0, y=0)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
plt.show()
