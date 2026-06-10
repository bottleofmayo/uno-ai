from pathlib import Path
import pandas as pd

df = pd.DataFrame({'file': [], 'line_count': []})
pd.options.display.max_rows = None

for file in Path('./results').iterdir():
    if file.is_file() and file.name != ".DS_Store":
        try:
            lines = file.read_text().splitlines()
        except UnicodeDecodeError as e:
            print("Error decoding in file:" + str(file))
        else: 
            # add row to df with file name and len(lines)
            df.loc[len(df)] = [file.name, len(lines)]

print("Files with 0 lines:")
print(len(df[df['line_count'] == 0]))

print("Files with 2 lines:")
print(len(df[df['line_count'] == 2]))

print("DMC files with 3 lines:")
print(len(df[(df['line_count'] == 3) & (df['file'].str.contains('dmc'))]))