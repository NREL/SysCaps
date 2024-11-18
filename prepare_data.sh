#!/bin/bash

# Download BuildingsBenh datasets
DATA_DIR=$1

cd $DATA_DIR

wget https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v2.0.0/compressed/BuildingsBench.tar.gz
wget https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v2.0.0/compressed/metadata.tar.gz
wget https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v2.0.0/compressed/resstock_amy2018.tar.gz
wget https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v2.0.0/compressed/resstock_tmy3.tar.gz
wget https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v2.0.0/compressed/comstock_amy2018.tar.gz
wget https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v2.0.0/compressed/comstock_tmy3.tar.gz

tar -xvf BuildingsBench.tar.gz
rm BuildingsBench.tar.gz

tar -xvf metadata.tar.gz
rm metadata.tar.gz

tar -xvf resstock_amy2018.tar.gz
rm resstock_amy2018.tar.gz

tar -xvf resstock_tmy3.tar.gz
rm resstock_tmy3.tar.gz

tar -xvf comstock_amy2018.tar.gz
rm comstock_amy2018.tar.gz

tar -xvf comstock_tmy3.tar.gz
rm comstock_tmy3.tar.gz

python -c "from huggingface_hub import snapshot_download; snapshot_download('NREL/SysCaps', local_dir='$SYSCAPS', repo_type='dataset')"

export SYSCAPS=$DATA_DIR/BuildingsBench