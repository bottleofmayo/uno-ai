import subprocess
from pathlib import Path
from joblib import Parallel, delayed  

models = [
    'random',
    'uno-rule-v1',
    'experiments/uno_dqn_result_large/model.pth',
    'experiments/uno_dqn_result_large2/model.pth',
    'experiments/uno_nfsp_result_large/model.pth',
    'experiments/uno_nfsp_result_large2/model.pth'
]

def evaluate(model1, model2, seed):
    process = subprocess.Popen(['python3', 'evaluate.py', '--models', model1, model2, '--seed', str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    file_path = Path(f"results/{model1.replace('experiments/','').replace('/model.pth','')}-{model2.replace('experiments/','').replace('/model.pth','')}-run{seed}.txt")
    file_path.write_text(stdout)
    return file_path

if __name__ == '__main__':  
    tasks = []
    large2_indices = [3, 5]  # Indices of large2 models
    for seed in range(1, 11):  # 10 runs
        for i in range(len(models)):
            for j in range(len(models)):
                # Only run if at least one model is large2
                if i in large2_indices or j in large2_indices:
                    tasks.append(delayed(evaluate)(models[i], models[j], seed))
    
    Parallel(n_jobs=-1)(tasks)