import subprocess
import os
import time

# List of languages to process.
languages = ['orm', 'sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul']
# 'amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug', 
# languages = ['amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul']
# Define available GPUs. (Assuming GPU ids 0 and 1)
available_gpus = [0, 1]

# List to keep track of running processes.
# Each entry is a tuple: (process, language, gpu)
running_processes = []

def launch_language(lang, gpu):
    """
    Launch dataset_creation.py for a given language on the specified GPU.
    It sets CUDA_VISIBLE_DEVICES so that the process uses only that GPU.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # It is assumed that your dataset_creation.py script has been updated to accept a '--lang' flag
    # so that it only processes the given language.
    cmd = ["python", "dataset_creation.py", "--lang", lang]
    process = subprocess.Popen(cmd, env=env)
    print(f"Launched language '{lang}' on GPU {gpu} (pid: {process.pid})")
    return process

def main():
    # Create an iterator over the languages.
    lang_iter = iter(languages)
    
    # Launch initial processes on all available GPUs.
    for gpu in available_gpus:
        try:
            lang = next(lang_iter)
            proc = launch_language(lang, gpu)
            running_processes.append((proc, lang, gpu))
        except StopIteration:
            break

    # Loop until all languages are processed.
    while running_processes:
        for i, (proc, lang, gpu) in enumerate(running_processes):
            ret = proc.poll()
            if ret is not None:  # process finished
                print(f"Language '{lang}' finished on GPU {gpu} with return code {ret}.")
                # Remove the finished process.
                running_processes.pop(i)
                # Launch the next language (if any) on the same GPU.
                try:
                    next_lang = next(lang_iter)
                    new_proc = launch_language(next_lang, gpu)
                    running_processes.append((new_proc, next_lang, gpu))
                except StopIteration:
                    # No more languages to process.
                    pass
                # Break out of the for-loop to re-check the running processes.
                break
        else:
            # No process finished yet; wait a bit before checking again.
            time.sleep(5)
    
    print("All languages processed.")

if __name__ == "__main__":
    main()
