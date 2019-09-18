# contrastive_sensor_fusion
(Soon to be) open-source code for the paper "Unsupervised Sensor Fusion With Contrastive Coding".

TODO(Aidan): sections on codebase organization, descriptions of each script, descriptions of arguments.


Experiments
-----------

Experiments in the paper were run with the following commands from this directory:

  python3 csf/scripts/degrading_dataset_experiment.py --flagfile=csf/parameters/experiments_classification.cfg

  python3 csf/scripts/degrading_dataset_experiment.py --flagfile=csf/parameters/experiments_classification.cfg --bands=PHR_red,PHR_green,PHR_blue

  python3 csf/scripts/degrading_dataset_experiment.py --flagfile=csf/parameters/experiments_classification.cfg --bands=PHR_red,PHR_green,PHR_blue --checkpoint_file=imagenet

  python3 csf/scripts/degrading_inputs_experiment.py --flagfile=csf/parameters/experiments_classification.cfg

  python3 csf/scripts/degrading_inputs_experiment.py --flagfile=csf/parameters/experiments_classification.cfg --bands=PHR_red,PHR_green,PHR_blue --checkpoint_file=imagenet
