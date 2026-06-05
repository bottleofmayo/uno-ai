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

df = pd.concat([df_dqn, df_nfsp], ignore_index=True)

"""
df['moving_avg'] = df['reward'].rolling(window=20).mean()
df_dqn['moving_avg'] = df_dqn['reward'].rolling(window=20).mean()
df_nfsp['moving_avg'] = df_nfsp['reward'].rolling(window=20).mean()
"""

#todo swap this for two seperate plots?
#sns.lineplot(data=df, x='episode', y='reward', hue='Algorithm')

sns.lineplot(data=df_dqn, x='episode', y='reward', label="DQN", errorbar="se")
sns.lineplot(data=df_nfsp, x='episode', y='reward', label="NFSP", errorbar="se")

#https://www.xkcd.com/2048/
sns.regplot(data=df_dqn, x='episode', y='reward', scatter=False, lowess=True, label="DQN Trend (LOWESS)")
sns.regplot(data=df_nfsp, x='episode', y='reward', scatter=False, lowess=True, label="NFSP Trend (LOWESS)")

plt.title("A Comparison of DQN and NFSP Reward During Training")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.savefig('outputs/training.png', dpi=600)
plt.savefig('outputs/training.svg')
