import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statistics
import math

df = pd.DataFrame()

for folder in Path('./experiments').iterdir():
    if folder.is_dir() and folder.name != "uno_dqn_result_large" and folder.name != "uno_nfsp_result_large":
        file = folder / 'performance.csv'
        df_new = pd.read_csv(file)
        df_new['algorithm'] = 'DQN' if 'dqn' in folder.name else 'NFSP'
        df = pd.concat([df, df_new])

df_dqn = df.loc[df['algorithm'] == 'DQN']
df_nfsp = df.loc[df['algorithm'] == 'NFSP']

#todo swap this for two seperate plots?
sns.lineplot(data=df, x='episode', y='reward', hue='algorithm', errorbar="se")

#https://www.xkcd.com/2048/
sns.regplot(data=df_dqn, x='episode', y='reward', scatter=False, lowess=True, label="DQN Trend (LOWESS)")
sns.regplot(data=df_nfsp, x='episode', y='reward', scatter=False, lowess=True, label="NFSP Trend (LOWESS)")

plt.title("DQN vs NFSP Reward During Training")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.legend()

plt.savefig('outputs/training_lesser.png', dpi=600)
plt.savefig('outputs/training_lesser.svg')

print(f"DQN end mean: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].mean()}")
print(f"DQN end median: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].median()}")
print(f"DQN end stdev: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].std()}")
print(f"DQN end sterr: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].std() / math.sqrt(len(df_dqn.loc[df_dqn['episode'] == 9800]))}")

print(f"nfsp end mean: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].mean()}")
print(f"nfsp end median: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].median()}")
print(f"nfsp end stdev: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].std()}")
print(f"nfsp end sterr: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].std() / math.sqrt(len(df_nfsp.loc[df_nfsp['episode'] == 9800]))}")