import subprocess
import itertools
import time
import threading
import queue
import psutil
import os
import signal
import itertools

# Declare the set of commands to run
methods = ["DORAEMON", "GOFLOW", "LSDR", "FullDR", "ADR", "NoDR"]
envs = ["Cartpole", "Humanoid", "Ant", "Quadcopter", "Anymal", "Gears"]

exp_name = "exp0_grand"

all_commands = []

hyperparams = None
# hyperparams = {
#     "GOFLOW": {
#         "alpha": [0.1, 0.5, 1.0, 1.5, 2.0],
#         # "beta": [0.0, 0.1, 0.5, 1.0, 2.0],    
#     },
#     # "DORAEMON": {
#     #     "kl_upper_bound": [0.005, 0.01, 0.05, 0.1, 0.5],
#     #     "success_rate_condition": [0.005, 0.01, 0.05, 0.1, 0.5],
#     # },
#     # "ADR": {
#     #     "th_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
#     # },
#     # "LSDR": {
#     #     "alpha": [0.1, 0.5, 1.0, 1.5, 2.0],
#     # }
# }

seeds = [0, 1, 2, 3, 4]

def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            os.kill(child.pid, signal.SIGKILL)
        os.kill(pid, signal.SIGKILL)
    except psutil.NoSuchProcess:
        pass

def kill_train_rl_processes():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.name().lower() and 'train_rl.py' in ' '.join(proc.cmdline()):
                print(f"Killing train_rl.py process with PID {proc.pid}")
                os.kill(proc.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def run_command_with_timeout(command, timeout=500):  # 5 minutes timeout
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    
    output_queue = queue.Queue()
    
    def enqueue_output(out, queue):
        for line in iter(out.readline, ''):
            queue.put(line)
        out.close()

    stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, output_queue))
    stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, output_queue))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    start_time = time.time()
    last_output_time = start_time
    last_wait_message = start_time

    try:
        while True:
            try:
                line = output_queue.get_nowait()
                print(line.strip())
                last_output_time = time.time()
            except queue.Empty:
                if process.poll() is not None:
                    return_code = process.poll()
                    print(f"Process terminated with return code: {return_code}")
                    if return_code != 0:
                        print(f"Process failed. Error code: {return_code}")
                    break  # Process has finished

                current_time = time.time()
                elapsed_time = current_time - last_output_time
                
                if elapsed_time > timeout:
                    print(f"Process not responding. Terminating due to {timeout} seconds of inactivity...")
                    kill_process_and_children(process.pid)
                    kill_train_rl_processes()
                    
                    print(f"Process terminated due to {timeout} seconds of inactivity")
                    return False
                
                if current_time - last_wait_message >= 10:
                    remaining_time = timeout - elapsed_time
                    print(f"Waiting... {remaining_time:.0f} seconds remaining before timeout")
                    last_wait_message = current_time
                
                time.sleep(0.1)  # Short sleep to prevent busy-waiting

        # Ensure we've processed all output
        while not output_queue.empty():
            print(output_queue.get_nowait().strip())

        return_code = process.returncode
        if return_code == 0:
            print("Process completed successfully")
        else:
            print(f"Process failed with return code: {return_code}")
        return return_code == 0

    except Exception as e:
        print(f"An error occurred while running the command: {str(e)}")
        return False

    finally:
        # Ensure process is terminated if we exit the function for any reason
        if process.poll() is None:
            print("Forcefully terminating the process...")
            kill_process_and_children(process.pid)
            kill_train_rl_processes()
            print("Process forcefully terminated")

if __name__ == "__main__":

    if(hyperparams is not None):

        for env in envs:
            for method, params in hyperparams.items():
                # Generate all combinations of key-value pairs
                keys, values = zip(*params.items())
                for combination in itertools.product(*values):
                    config_overrides = ",".join(
                        f"dr_method.{key}={value}" for key, value in zip(keys, combination)
                    )
                    command = (
                        f"python -u train_rl.py --task={env}-{method}-v0 --headless "
                        f"--exp_name={exp_name} --config_overrides \"{config_overrides}\""
                    )
                    all_commands.append(command)
    else:
        for env in envs:
            all_commands.extend([f"python -u train_rl.py --task={env}-{method}-v0 --headless --exp_name={exp_name}" for method in methods])



    for seed in seeds:
        for command in all_commands:
            command = command + f" --seed={seed}"
            print(command)
            print(f"Running command: {command}")
            
            max_retries = 5
            for attempt in range(max_retries):
                if run_command_with_timeout(command):
                    print(f"Command completed successfully: {command}")
                    break
                else:
                    print(f"Attempt {attempt + 1} failed. Retrying...")
            else:
                print(f"Command failed after {max_retries} attempts: {command}")
            
            print("---")

    print("All commands completed.")
