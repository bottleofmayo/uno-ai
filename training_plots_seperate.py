import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#todo does this count as plotting raw data?

# Load data
df_dqn = pd.read_csv("experiments/uno_dqn_result_large/performance.csv")
df_dqn["Algorithm"] = "DQN"

df_nfsp = pd.read_csv("experiments/uno_nfsp_result_large/performance.csv")
df_nfsp["Algorithm"] = "NFSP"

# Initialize a stacked subplot layout (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

# --- Top Subplot: DQN ---
sns.lineplot(data=df_dqn, x="episode", y="reward", ax=axes[0], label="Raw Reward")
sns.regplot(
    data=df_dqn,
    x="episode",
    y="reward",
    scatter=False,
    lowess=True,
    ax=axes[0],
    label="Trend (LOWESS)",
)
axes[0].set_title("DQN Performance")
axes[0].set_ylabel("Reward")
axes[0].set_xlabel("")  # Hidden because sharex=True shares the bottom x-axis
axes[0].legend()

# --- Bottom Subplot: NFSP ---
sns.lineplot(
    data=df_nfsp, x="episode", y="reward", ax=axes[1], label="Raw Reward"
)
sns.regplot(
    data=df_nfsp,
    x="episode",
    y="reward",
    scatter=False,
    lowess=True,
    ax=axes[1],
    label="Trend (LOWESS)",
)
axes[1].set_title("NFSP Performance")
axes[1].set_ylabel("Reward")
axes[1].legend()

# Optimize spacing between graphs
plt.tight_layout()
plt.show()
