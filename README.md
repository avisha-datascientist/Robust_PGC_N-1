# Robust_PGC_N-1

This is the official repository for the paper **Jacobian-Based Adversarial Attacks for N-1
Robust Power Grid Control**.

# Credit:
This codebase uses and is based on Semi-Markov Afterstate Actor Critic agent from https://github.com/sunghoonhong/SMAAC/tree/master.

# Environment:
python - 3.7.10
grid2op - 1.9.0
lightsim2grid - 0.7.5
cuda - 12.4.1

# Train:
To train the general format is:

python -u /work/smavbhir/project_thesis/SMAAC/test.py -c=environment_name -s=seed -n=run_name -rn=Run_number -bs=batch_size

Examples:
For WCCI:
python -u /work/smavbhir/project_thesis/SMAAC/test.py -c=wcci -n=casewcci_100k_rn1_01_train -rn=5 -bs=64 -s=0

For other grids:
python -u /work/smavbhir/project_thesis/SMAAC/test.py -c=5 -n=case5_100k_rn1_01_train -rn=5 -s=0

# Evaluation:

For evaluation the general format is:

python -u /work/smavbhir/project_thesis/SMAAC/evaluate.py -c=environment_name -n=run_name[case{environment_name}_{steps}_rn1_{epsilon}_train_{seed_number}]

WCCI, 14 and 5:
python -u /work/smavbhir/project_thesis/SMAAC/evaluate.py -c=5 -n=case5_100k_rn1_0_train_0
NeurIPS:

python -u /work/smavbhir/project_thesis/SMAAC/evaluate_neurips.py -c=neur_ips -n=case5_100k_rn1_0_train_0

# Data folder:
Contains data of only two grids. Data of any new grid used for the first time will be downloaded here.

# Results folder:
Contains trained models on 3 grids namely case5, case14 and wcci.

