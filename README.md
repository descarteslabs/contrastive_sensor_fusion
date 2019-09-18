# contrastive_sensor_fusion
(Soon to be) open-source code for the paper "Unsupervised Sensor Fusion With Contrastive Coding".

TODO(Aidan): sections on codebase organization, descriptions of each script, descriptions of arguments.


Experiments
-----------

Experiments in the paper were run with the following commands from this directory:

    export EXPERIMENT='python3 csf/scripts/classification_experiment.py --flagfile=csf/parameters/experiments_classification.cfg'
    export OURS='***REMOVED***'

    # Experiments which drop bands in order:
    $EXPERIMENT --checkpoint_file=$OURS # (all bands)
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red,NAIP_green,NAIP_blue,NAIP_nir,PHR_red,PHR_green,PHR_blue
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red,NAIP_green,NAIP_blue,NAIP_nir,PHR_red,PHR_green
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red,NAIP_green,NAIP_blue,NAIP_nir,PHR_red
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red,NAIP_green,NAIP_blue,NAIP_nir
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red,NAIP_green,NAIP_blue
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red,NAIP_green
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red
  
    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red,PHR_green,PHR_blue
    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red,PHR_green
    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red

    # Experiments which shrink the dataset:
    $EXPERIMENT --checkpoint_file=$OURS --dataset_size=8000
    $EXPERIMENT --checkpoint_file=$OURS --dataset_size=5000
    $EXPERIMENT --checkpoint_file=$OURS --dataset_size=2000
    $EXPERIMENT --checkpoint_file=$OURS --dataset_size=1000
    $EXPERIMENT --checkpoint_file=$OURS --dataset_size=500
  
    $EXPERIMENT --checkpoint_file=$OURS --bands=PHR_red,PHR_green,PHR_blue --dataset_size=8000
    $EXPERIMENT --checkpoint_file=$OURS --bands=PHR_red,PHR_green,PHR_blue --dataset_size=5000
    $EXPERIMENT --checkpoint_file=$OURS --bands=PHR_red,PHR_green,PHR_blue --dataset_size=2000
    $EXPERIMENT --checkpoint_file=$OURS --bands=PHR_red,PHR_green,PHR_blue --dataset_size=1000
    $EXPERIMENT --checkpoint_file=$OURS --bands=PHR_red,PHR_green,PHR_blue --dataset_size=500

    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red,PHR_green,PHR_blue --dataset_size=8000
    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red,PHR_green,PHR_blue --dataset_size=5000
    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red,PHR_green,PHR_blue --dataset_size=2000
    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red,PHR_green,PHR_blue --dataset_size=1000
    $EXPERIMENT --checkpoint_file=imagenet --bands=PHR_red,PHR_green,PHR_blue --dataset_size=500

    # Experiments which drop specific bands:
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir # (redundant)
    $EXPERIMENT --checkpoint_file=$OURS --bands=NAIP_red,NAIP_green,NAIP_blue,NAIP_nir
    $EXPERIMENT --checkpoint_file=$OURS --bands=PHR_red,PHR_green,PHR_blue,PHR_nir
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,NAIP_red,NAIP_green,NAIP_blue,NAIP_nir
    $EXPERIMENT --checkpoint_file=$OURS --bands=NAIP_red,NAIP_green,NAIP_blue,NAIP_nir,PHR_red,PHR_green,PHR_blue,PHR_nir
    $EXPERIMENT --checkpoint_file=$OURS --bands=SPOT_red,SPOT_green,SPOT_blue,SPOT_nir,PHR_red,PHR_green,PHR_blue,PHR_nir
