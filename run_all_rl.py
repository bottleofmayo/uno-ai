import subprocess

subprocess.Popen(['python3', 'run_rl.py', '--log_dir', 'experiments/uno_dqn_result_large2', '--num_episodes', '100000', '--seed', '43'])
subprocess.Popen(['python3', 'run_rl.py', '--log_dir', 'experiments/uno_nfsp_result_large2', '--algorithm', 'nfsp', '--num_episodes', '100000', '--seed', '43'])