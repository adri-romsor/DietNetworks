                #!/bin/sh

                ## Reasonable default values
                # Execute the job from the current working directory.
                #PBS -d /home/tisu32/feature_selection

                #All jobs must be submitted with an estimated run time
                #PBS -l walltime=23:59:59

                ## Job name
                #PBS -N dbi_10f4e484fd9

                ## log out/err files
                # We cannot use output_file and error_file here for now.
                # We will use dbi_...out-id and dbi_...err-id instead
                # Torque append -{JOBID} to the output filename
                # But not Moab, so we add it
                #PBS -o /home/tisu32/feature_selection/LOGS/python_-u__home_tisu32_feature_selection_experiments_variant2_learn_gene_vector.py_-lr_.0001_-a_.5_-b_.5_2016-10-30_11-59-13.942623/dbi_10f4e484fd9.out-${MOAB_JOBARRAYINDEX}
                #PBS -e /home/tisu32/feature_selection/LOGS/python_-u__home_tisu32_feature_selection_experiments_variant2_learn_gene_vector.py_-lr_.0001_-a_.5_-b_.5_2016-10-30_11-59-13.942623/dbi_10f4e484fd9.err-${MOAB_JOBARRAYINDEX}


                ## The project name for accounting/permission
                #PBS -A jvb-000-ag

        ## Number of CPU (on the same node) per job
        #PBS -l nodes=1:gpus=1

                ## Execute as many jobs as needed
                #PBS -t 0-0%30

                ## Queue name
                #PBS -q gpu_1
export "THEANO_FLAGS=device=gpu"
export "JOBDISPATCH_RESUBMIT=msub /home/tisu32/feature_selection/LOGS/python_-u__home_tisu32_feature_selection_experiments_variant2_learn_gene_vector.py_-lr_.0001_-a_.5_-b_.5_2016-10-30_11-59-13.942623/submit.sh"

                ## Variable to put into the environment
                #PBS -v THEANO_FLAGS,JOBDISPATCH_RESUBMIT

                ## Execute the 'launcher' script in bash
                # Bash is needed because we use its "array" data structure
                # the -l flag means it will act like a login shell,
                # and source the .profile, .bashrc, and so on
                /bin/bash -l -e /home/tisu32/feature_selection/LOGS/python_-u__home_tisu32_feature_selection_experiments_variant2_learn_gene_vector.py_-lr_.0001_-a_.5_-b_.5_2016-10-30_11-59-13.942623/launcher
