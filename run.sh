#!/bin/bash
set -e  

cd study1/nudge_replication

echo "Running Study 1 - Short-Term Experiments: run.py"
python run.py 

echo "Running Study 1 - Long-Term Experiments: run_long_term_0.py (Allcott)"
python run_long_term_0.py  

echo "Running Study 1 - Long-Term Experiments: run_long_term_2.py (Paunov)"
python run_long_term_2.py

echo "Running Study 1 - Long-Term Experiments: run_long_term_3.py (Vivek)"
python run_long_term_3.py

cd ../../study2/longitudinal_simulation

echo "Running Study 2 - run_long.py (one nudge)"
python run_long.py

echo "Running Study 2 - run_freq.py (scheduled nudges)"
python run_freq.py

cd ../../study4/social_simulation

echo "Running Study 4 - run_contact.py"
python run_contact.py

echo "All studies finished successfully!"
