from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

df = pd.DataFrame({'P1': [], 'P2': [], 'P1 Score': [], 'P2 Score': []})

replacements = {
    "experiments/uno_dqn_result_large/model.pth": "DQN",
    "experiments/uno_dqn_result_large2/model.pth": "DQN",
    "experiments/uno_nfsp_result_large/model.pth": "NFSP",
    "experiments/uno_nfsp_result_large2/model.pth": "NFSP",
    "experiments/uno_dmc_result_large/model.pth": "DMC",
    "experiments/uno_dmc_result_large2/model.pth": "DMC",
    "random": "Random",
    "uno-rule-v1": "Rule Model"
}

for file in Path('./results').iterdir():
    if file.is_file() and file.name != ".DS_Store":
        try:
            lines = file.read_text().splitlines()
        except UnicodeDecodeError as e:
            print("Error decoding in file:" + str(file))
            
        else:
            if len(lines) == 0:
                print("Empty file:" + str(file))
            else:
                offset = 0
                if len(lines) == 2:
                    print("File with only 2 lines:" + str(file))
                    offset = -1
                try:
                    p1_name = lines[1 + offset].split()[1]
                    p1_name = replacements.get(p1_name, p1_name)
                    p2_name = lines[2 + offset].split()[1]
                    p2_name = replacements.get(p2_name, p2_name)
                    p1_score = lines[1 + offset].split()[-1]
                    p2_score = lines[2 + offset].split()[-1]
                    df.loc[len(df)] = [p1_name, p2_name, p1_score, p2_score]
                except IndexError as e:
                    print("Error parsing file:" + str(file))
                    print("File content:")
                    print(lines)

df['P1 Score'] = pd.to_numeric(df['P1 Score'], errors='coerce')                
df['P2 Score'] = pd.to_numeric(df['P2 Score'], errors='coerce')

df.to_csv('outputs/data.csv')

print(f"DQN mean: {df.loc[df['P1'] == 'DQN', 'P1 Score'].mean()}")
print(f"NFSP mean: {df.loc[df['P1'] == 'NFSP', 'P1 Score'].mean()}")
print(f"DMC mean: {df.loc[df['P1'] == 'DMC', 'P1 Score'].mean()}")
print(f"DQN std: {df.loc[df['P1'] == 'DQN', 'P1 Score'].std()}")
print(f"NFSP std: {df.loc[df['P1'] == 'NFSP', 'P1 Score'].std()}")
print(f"DMC std: {df.loc[df['P1'] == 'DMC', 'P1 Score'].std()}")
print(f"DQN std error: {df.loc[df['P1'] == 'DQN', 'P1 Score'].std()/math.sqrt(len(df.loc[df['P1'] == 'DQN', 'P1 Score']))}")
print(f"NFSP std error: {df.loc[df['P1'] == 'NFSP', 'P1 Score'].std()/math.sqrt(len(df.loc[df['P1'] == 'NFSP', 'P1 Score']))}")
print(f"DMC std error: {df.loc[df['P1'] == 'DMC', 'P1 Score'].std()/math.sqrt(len(df.loc[df['P1'] == 'DMC', 'P1 Score']))}")

df_pivot = df.pivot_table(index='P1', columns="P2", values="P1 Score")
df_pivot_se = df.pivot_table(index='P1', columns="P2", values="P1 Score", aggfunc='sem')

annot_matrix = df_pivot.map(lambda x: f"{x:.4f}") + ' ± ' + df_pivot_se.map(lambda x: f"{x:.4f}")

print(df_pivot_se)

ax = sns.heatmap(df_pivot, annot=annot_matrix, fmt="", annot_kws={"size": 6})
ax.set(title="Player 1 Win Rate Across Different Model Matchups", xlabel="Player 2", ylabel="Player 1")
plt.savefig('outputs/heatmap.png', dpi=600)
plt.savefig('outputs/heatmap.svg')

