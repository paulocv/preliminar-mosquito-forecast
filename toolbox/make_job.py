"""SCRIPT TO PREPARE A .job FILE

Usage
-----
- Input file as first argument
- Output folder as second argument
"""

import os
import sys
from toolbox.file_tools import get_folder_name_from_argv, get_bool_from_dict, \
    read_config_file, read_optional_from_dict

STD_SELECT = 1  # Standard number of nodes
STD_NCPUS = 40  # Standard number of CPUs. 40 or 46
STD_WALLTIME = "150:00:00"  # Standard walltime. 0 to 24 to 336.

MAIN_FOLDER = "/home/paulocvs/Moving_agents"
VIRTUALENV = "/home/paulocvs/full_python3/bin/activate"
JOB_OUT_FOLDER = MAIN_FOLDER + "jobs/out/"
JOB_ERR_FOLDER = MAIN_FOLDER + "jobs/err/"

STD_JOBNAME = "jobs/std.job"
MAIL_ADDRESS = "paulo.ventura.silva@usp.br"

STD_RUN_SCRIPT = "montec_paramvar.py"


def main():
    # Gets the multinput file name and reads it
    input_file = sys.argv[1]
    input_dict = read_config_file(input_file)

    # Gets output folder and creates it, if needed
    output_folder = get_folder_name_from_argv(argi=2)

    # ----------------------------
    # Job parameters
    commands = ""

    # Job ID
    commands += "#PBS -N {}\n".format(input_dict["file_prefix"])

    # Number of nodes
    try:
        select = input_dict["select"]
    except KeyError:
        select = STD_SELECT
    commands += "#PBS -l select={}".format(select)

    # Number of CPUs
    try:
        ncpus = input_dict["ncpus"]
    except KeyError:
        ncpus = STD_NCPUS
    commands += ":ncpus={}\n".format(ncpus)

    # Walltime
    try:
        walltime = input_dict["walltime"]
    except KeyError:
        walltime = STD_WALLTIME
    commands += "#PBS -l walltime={}\n\n".format(walltime)

    # Output and error files folders
    commands += "#PBS -o {}\n".format(JOB_OUT_FOLDER)
    commands += "#PBS -e {}\n\n".format(JOB_ERR_FOLDER)

    # Mail information (false if key is not found)
    mail_bool = get_bool_from_dict(input_dict, "send_mail")
    if mail_bool:
        commands += "#PBS -m abe\n"
        commands += "#PBS -M {}\n\n".format(MAIL_ADDRESS)

    # Script to run
    run_script = read_optional_from_dict(input_dict, "run_script",
                                         STD_RUN_SCRIPT)

    # ----------------------
    # Job commands
    commands += "source {}\n".format(VIRTUALENV)
    commands += "cd {}\n\n".format(MAIN_FOLDER)

    commands += "python " + run_script + " " + input_file + " " + output_folder + "\n"

    # ----------------
    # User feedback and interaction

    print("--------------")
    print("Run script: {}".format(run_script))
    # print("Model: {}".format(input_dict["model"]))
    print("Job ID: {}".format(input_dict["file_prefix"]))

    print("Job parameters:")
    print("  |select = {}".format(select))
    print("  |ncpus =  {}".format(ncpus))
    print("  |walltime = {}".format(walltime))
    print("Send mail ({}) = {}".format(mail_bool, MAIL_ADDRESS))

    job_name = STD_JOBNAME
    while True:
        print("\nJob file name: {}".format(job_name))
        ans = input("Submit job now? (y/n. r = rename job file.)")
        if ans == "y":
            submit = True
            break
        elif ans == "n":
            submit = False
            break
        elif ans == "r":
            job_name = input("New job name: ")

    # Job file writing
    with open(job_name, "w") as fp:
        fp.write(commands)

    # Submit
    if submit:
        os.system("qsub " + job_name)


if __name__ == "__main__":
    main()
