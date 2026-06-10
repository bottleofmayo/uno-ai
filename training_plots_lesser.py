import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statistics
import math

df = pd.DataFrame()

for folder in Path('./experiments').iterdir():
    if folder.is_dir() and not 'large' in folder.name:
        file = folder / 'performance.csv'
        df_new = pd.read_csv(file)
        df_new['algorithm'] = 'DQN' if 'dqn' in folder.name else 'DMC' if 'dmc' in folder.name else 'NFSP'
        df = pd.concat([df, df_new])

df_dqn = df.loc[df['algorithm'] == 'DQN']
df_nfsp = df.loc[df['algorithm'] == 'NFSP']
df_dmc = df.loc[df['algorithm'] == 'DMC']

#todo swap this for two seperate plots?
sns.lineplot(data=df, x='episode', y='reward', hue='algorithm', errorbar="se")

#https://www.xkcd.com/2048/
sns.regplot(data=df_dqn, x='episode', y='reward', scatter=False, lowess=True, label="DQN Trend (LOWESS)")
sns.regplot(data=df_nfsp, x='episode', y='reward', scatter=False, lowess=True, label="NFSP Trend (LOWESS)")
sns.regplot(data=df_dmc, x='episode', y='reward', scatter=False, lowess=True, label="DMC Trend (LOWESS)")

plt.title("A Comparison of DQN, NFSP and DMC Reward During Training \n over 10,000 Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.legend()

plt.savefig('outputs/training_lesser.png', dpi=600)
plt.savefig('outputs/training_lesser.svg')

print(f"DQN end mean: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].mean()}")
print(f"DQN end median: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].median()}")
print(f"DQN end stdev: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].std()}")
print(f"DQN end sterr: {df_dqn.loc[df_dqn['episode'] == 9800, 'reward'].std() / math.sqrt(len(df_dqn.loc[df_dqn['episode'] == 9800]))}")

print(f"NFSP end mean: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].mean()}")
print(f"NFSP end median: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].median()}")
print(f"NFSP end stdev: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].std()}")
print(f"NFSP end sterr: {df_nfsp.loc[df_nfsp['episode'] == 9800, 'reward'].std() / math.sqrt(len(df_nfsp.loc[df_nfsp['episode'] == 9800]))}")

print(f"DMC end mean: {df_dmc.loc[df_dmc['episode'] == 9800, 'reward'].mean()}")
print(f"DMC end median: {df_dmc.loc[df_dmc['episode'] == 9800, 'reward'].median()}")
print(f"DMC end stdev: {df_dmc.loc[df_dmc['episode'] == 9800, 'reward'].std()}")
print(f"DMC end sterr: {df_dmc.loc[df_dmc['episode'] == 9800, 'reward'].std() / math.sqrt(len(df_dmc.loc[df_dmc['episode'] == 9800]))}")