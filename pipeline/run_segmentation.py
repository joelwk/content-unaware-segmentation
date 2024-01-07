import sys
from importlib import import_module

def run_script(script_main_func, script_name):
    try:
        status = script_main_func()
        print(f"Debug: {script_name} returned status {status} of type {type(status)}") # Debug statement - keeping for information output
        if status != 0:
            print(f"Error in {script_name}, stopping pipeline.")
            return status
    except Exception as e:
        print(f"An exception occurred in {script_name}: {e}")
        return 1
    return None  

def run():
    scripts = [
        ("main", "pipeline"),
        ("main", "getvideo2dataset"),
        ("main", "clipvideoencode"),
        ("initialize_and_run", "segment"),]
    for idx, (script_func_name, script_path) in enumerate(scripts):
        print(f"Running script: {script_path}.{script_func_name}")  # Debug statement - keeping for information output
        script_func = import_module(script_path).__dict__[script_func_name] 
        exit_status = run_script(script_func, script_func_name)
        if exit_status is not None and exit_status != 0:
            print(f"Exiting due to error in script: {script_path}.{script_func_name}")  # Debug statement - keeping for information output
            sys.exit(exit_status)
if __name__ == "__main__":
    run()


    