from numpy import shape
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Found {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(f"⚠️ GPU initialization error: {e}")
else:
    print("⚙️ No GPU available, running on CPU.")



df = pd.read_csv('Football_Player_Data-Analysis.csv')

def map_position(pos: str):
    if pd.isna(pos) or pos == '':
        return None
    pos = pos.upper()
    if 'FW' in pos:
        return 'Forward'
    elif 'AM' in pos or 'M' in pos:
        return 'Midfielder'
    elif 'D' in pos:
        return 'Defender'
    else:
        return None


#print(df.info())


df['Main Position 1'] = df['Position 1'].apply(map_position)
df['Main Position 2'] = df['Position 2'].apply(map_position)

df['Main Position'] = df['Main Position 1'].combine_first(df['Main Position 2'])

df_FW = df[df['Main Position'] == 'Forward']
df_CM = df[df['Main Position'] == 'Midfielder']
df_CB = df[df['Main Position'] == 'Defender']

df_FW = df_FW.fillna(0)
df_CM = df_CM.fillna(0)
df_CB = df_CB.fillna(0)

x_train_FW = ['Player Age','Mins Played', 'shotsPerGame', 'dribbleWonPerGame'] 
y_train_FW = ['goal', 'assistTotal', 'rating']

x_train_CM = ['Player Age', 'Mins Played', 'passSuccess', 'totalPassesPerGame', 'accurateCrossesPerGame']
y_train_CM = ['assistTotal', 'keyPassPerGame', 'rating',] 

x_train_CB = ['Player Age','Mins Played', 'interceptionPerGame', 'foulGivenPerGame', 'wasDribbledPerGame', 'totalPassesPerGame', 'accurateLongPassPerGame']
y_train_CB = ['tacklePerGame', 'clearancePerGame', 'rating']

X_FW = df_FW[x_train_FW]
Y_FW = df_FW[y_train_FW]

X_CM = df_CM[x_train_CM]
Y_CM = df_CM[y_train_CM]

X_CB = df_CB[x_train_CB]
Y_CB = df_CB[y_train_CB]

scaler_FW = StandardScaler()
X_FW_scaled = scaler_FW.fit_transform(X_FW)

scaler_CM = StandardScaler()
X_CM_scaled = scaler_CM.fit_transform(X_CM)

scaler_CB = StandardScaler()
X_CB_scaled = scaler_CB.fit_transform(X_CB)


def train_model(X, y):
    model = Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(y.shape[1])
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=200, verbose=0)
    return model
                       

Y_FW_array = Y_FW.values
Y_CM_array = Y_CM.values
Y_CB_array = Y_CB.values



model_FW = train_model(X_FW_scaled, Y_FW_array)
model_CM = train_model(X_CM_scaled, Y_CM_array)
model_CB = train_model(X_CB_scaled, Y_CB_array)

def make_results(df_orig, y_pred, y_true_cols):
    df_res = pd.DataFrame(y_pred, columns=[f"Predicted {col}" for col in y_true_cols])
    df_res["Player Name"] = df_orig["Player Name"].values
    df_res["Player Team"] = df_orig["Player Team"].values
    df_res["Main Position"] = df_orig["Position 1"].values
    return df_res

FW_pred = model_FW.predict(X_FW_scaled)
CM_pred = model_CM.predict(X_CM_scaled)
CB_pred = model_CB.predict(X_CB_scaled)

df_FW_results = make_results(df_FW, FW_pred, y_train_FW)
df_CM_results = make_results(df_CM, CM_pred, y_train_CM)
df_CB_results = make_results(df_CB, CB_pred, y_train_CB)


df_results_all = pd.concat([df_FW_results, df_CM_results, df_CB_results], ignore_index=True)


df_team_stats = df_results_all.groupby("Player Team").agg({
    "Predicted goal": "sum",
    "Predicted assistTotal": "sum",
    "Predicted rating": "mean"
}).reset_index()

df_team_stats['Team Power'] = (
    df_team_stats['Predicted goal'] * 0.6 +
    df_team_stats['Predicted assistTotal'] * 0.4 +
    df_team_stats['Predicted rating'] * 10
)

df_team_stats = df_team_stats.sort_values(by='Team Power', ascending=False).reset_index(drop=True)


print("=== Statystyki drużynowe (Top 10) ===")
print(df_team_stats.head(10))
print("\n")




# --- TOP 10 napastników ---
top_FW = (
    df_FW_results
    .sort_values(by='Predicted rating', ascending=False)
    .head(10)
    [['Player Name', 'Player Team', 
      'Predicted goal',
      'Predicted assistTotal', 
      'Predicted rating']]
)
print("🔝 Top 10 napastników:")
print(top_FW)

# --- TOP 10 pomocników ---
top_CM = (
    df_CM_results
    .sort_values(by='Predicted rating', ascending=False)
    .head(10)
    [['Player Name', 'Player Team',
      'Predicted keyPassPerGame',
      'Predicted assistTotal', 
      'Predicted rating']]
)


print("\n🎯 Top 10 pomocników:")
print(top_CM)

# --- TOP 10 obrońców ---
top_CB = (
    df_CB_results
    .sort_values(by='Predicted rating', ascending=False) 
    .head(10)
    [['Player Name', 'Player Team',
      'Predicted tacklePerGame',
      'Predicted clearancePerGame',
      'Predicted rating',]]
)

pd.set_option('display.max_rows', None)
print("\n🛡️ Top 10 obrońców:")
print(top_CB)
