import subprocess

for i in range(10):
    subprocess.Popen(['python3', 'run_rl.py', '--log_dir', f"experiments/uno_dqn_result_{i+1}", '--num_episodes', '10000', "--evaluate_every", "100", "--num_eval_games", "2000", "--seed", f"{i+1}"])
    subprocess.Popen(['python3', 'run_rl.py', '--log_dir', f"experiments/uno_nfsp_result_{i+1}", '--algorithm', 'nfsp', '--num_episodes', '10000', "--evaluate_every", "100", "--num_eval_games", "2000", "--seed", f"{i+1}"])
    subprocess.Popen(['python3', 'run_dmc.py', '--log_dir', f"experiments/uno_dmc_result_{i+1}", '--num_episodes', '10000', '--frames_per_episode', '100', "--evaluate_every", "100", "--num_eval_games", "2000", "--seed", f"{i+1}"])

subprocess.Popen(['python3', 'run_rl.py', '--log_dir', 'experiments/uno_dqn_result_large', '--num_episodes', '100000'])
subprocess.Popen(['python3', 'run_rl.py', '--log_dir', 'experiments/uno_nfsp_result_large', '--algorithm', 'nfsp', '--num_episodes', '100000'])
subprocess.Popen(['python3', 'run_dmc.py', '--log_dir', 'experiments/uno_dmc_result_large', '--num_episodes', '100000', '--frames_per_episode', '100'])
subprocess.Popen(['python3', 'run_rl.py', '--log_dir', 'experiments/uno_dqn_result_large2', '--num_episodes', '100000', '--seed', '43'])
subprocess.Popen(['python3', 'run_rl.py', '--log_dir', 'experiments/uno_nfsp_result_large2', '--algorithm', 'nfsp', '--num_episodes', '100000', '--seed', '43'])
subprocess.Popen(['python3', 'run_dmc.py', '--log_dir', 'experiments/uno_dmc_result_large2', '--num_episodes', '100000', '--frames_per_episode', '100', '--seed', '43'])