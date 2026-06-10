import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df_dqn_1 = pd.read_csv('experiments/uno_dqn_result_large/performance.csv')
df_dqn_2 = pd.read_csv('experiments/uno_dqn_result_large2/performance.csv')
df_dqn = pd.concat([df_dqn_1, df_dqn_2])
df_dqn['Algorithm'] = 'DQN'

df_nfsp_1 = pd.read_csv('experiments/uno_nfsp_result_large/performance.csv')
df_nfsp_2 = pd.read_csv('experiments/uno_nfsp_result_large2/performance.csv')
df_nfsp = pd.concat([df_nfsp_1, df_nfsp_2])
df_nfsp['Algorithm'] = 'NFSP'

df_dmc_1 = pd.read_csv('experiments/uno_dmc_result_large/performance.csv')
df_dmc_2 = pd.read_csv('experiments/uno_dmc_result_large2/performance.csv')
df_dmc = pd.concat([df_dmc_1, df_dmc_2])
df_dmc['Algorithm'] = 'DMC'

df = pd.concat([df_dqn, df_nfsp, df_dmc], ignore_index=True)

"""
df['moving_avg'] = df['reward'].rolling(window=20).mean()
df_dqn['moving_avg'] = df_dqn['reward'].rolling(window=20).mean()
df_nfsp['moving_avg'] = df_nfsp['reward'].rolling(window=20).mean()
"""

#todo swap this for two seperate plots?
#sns.lineplot(data=df, x='episode', y='reward', hue='Algorithm')

sns.lineplot(data=df_dqn, x='episode', y='reward', label="DQN", errorbar="se")
sns.lineplot(data=df_nfsp, x='episode', y='reward', label="NFSP", errorbar="se")
sns.lineplot(data=df_dmc, x='episode', y='reward', label="DMC", errorbar="se")

#https://www.xkcd.com/2048/
sns.regplot(data=df_dqn, x='episode', y='reward', scatter=False, lowess=True, label="DQN Trend (LOWESS)")
sns.regplot(data=df_nfsp, x='episode', y='reward', scatter=False, lowess=True, label="NFSP Trend (LOWESS)")
sns.regplot(data=df_dmc, x='episode', y='reward', scatter=False, lowess=True, label="DMC Trend (LOWESS)")

plt.title("A Comparison of DQN, NFSP and DMC Reward During Training \n over 100,000 Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

plt.savefig('outputs/training.png', dpi=600)
plt.savefig('outputs/training.svg')

print(f"DQN end mean: {df_dqn.loc[df_dqn['episode'] == 99800, 'reward'].mean()}")
print(f"DQN end median: {df_dqn.loc[df_dqn['episode'] == 99800, 'reward'].median()}")
print(f"DQN end stdev: {df_dqn.loc[df_dqn['episode'] == 99800, 'reward'].std()}")
print(f"DQN end sterr: {df_dqn.loc[df_dqn['episode'] == 99800, 'reward'].std() / math.sqrt(len(df_dqn.loc[df_dqn['episode'] == 99800]))}")

print(f"NFSP end mean: {df_nfsp.loc[df_nfsp['episode'] == 99800, 'reward'].mean()}")
print(f"NFSP end median: {df_nfsp.loc[df_nfsp['episode'] == 99800, 'reward'].median()}")
print(f"NFSP end stdev: {df_nfsp.loc[df_nfsp['episode'] == 99800, 'reward'].std()}")
print(f"NFSP end sterr: {df_nfsp.loc[df_nfsp['episode'] == 99800, 'reward'].std() / math.sqrt(len(df_nfsp.loc[df_nfsp['episode'] == 99800]))}")

print(f"DMC end mean: {df_dmc.loc[df_dmc['episode'] == 99800, 'reward'].mean()}")
print(f"DMC end median: {df_dmc.loc[df_dmc['episode'] == 99800, 'reward'].median()}")
print(f"DMC end stdev: {df_dmc.loc[df_dmc['episode'] == 99800, 'reward'].std()}")
print(f"DMC end sterr: {df_dmc.loc[df_dmc['episode'] == 99800, 'reward'].std() / math.sqrt(len(df_dmc.loc[df_dmc['episode'] == 99800]))}")