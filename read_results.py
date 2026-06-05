from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({'P1': [], 'P2': [], 'P1 Score': [], 'P2 Score': []})

for file in Path('./results').iterdir():
    if file.is_file() and file.name != ".DS_Store":
        try:
            lines = file.read_text().splitlines()
        except UnicodeDecodeError as e:
            print("Error decoding in file:" + str(file))
            
        else:
            p1_name = lines[1].split()[1].replace("experiments/uno_dqn_result_large/model.pth", "DQN").replace("experiments/uno_dqn_result_large2/model.pth", "DQN").replace("experiments/uno_nfsp_result_large/model.pth", "NFSP").replace("experiments/uno_nfsp_result_large2/model.pth", "NFSP").replace("random", "Random").replace("uno-rule-v1", "Rule Model")
            p2_name = lines[2].split()[1].replace("experiments/uno_dqn_result_large/model.pth", "DQN").replace("experiments/uno_dqn_result_large2/model.pth", "DQN").replace("experiments/uno_nfsp_result_large/model.pth", "NFSP").replace("experiments/uno_nfsp_result_large2/model.pth", "NFSP").replace("random", "Random").replace("uno-rule-v1", "Rule Model")
            p1_score = lines[1].split()[-1]
            p2_score = lines[2].split()[-1]
            df.loc[len(df)] = [p1_name, p2_name, p1_score, p2_score]

df['P1 Score'] = pd.to_numeric(df['P1 Score'], errors='coerce')                
df['P2 Score'] = pd.to_numeric(df['P2 Score'], errors='coerce')

df.to_csv('outputs/data.csv')

df_pivot = df.pivot_table(index='P1', columns="P2", values="P1 Score")
df_pivot_se = df.pivot_table(index='P1', columns="P2", values="P1 Score", aggfunc='sem')

annot_matrix = df_pivot.map(lambda x: f"{x:.4f}") + ' ± ' + df_pivot_se.map(lambda x: f"{x:.4f}")

print(df_pivot_se)

ax = sns.heatmap(df_pivot, annot=annot_matrix, fmt="", annot_kws={"size": 6})
ax.set(title="Player 1 Win Rate Across Different Model Matchups", xlabel="Player 2", ylabel="Player 1")
plt.savefig('outputs/heatmap.png', dpi=600)
plt.savefig('outputs/heatmap.svg')

