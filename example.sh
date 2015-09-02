#!/usr/bin/env bash
#
# Example of running the SSCM algorithm.
#

# Download the paper's dataset.
echo 'Fetching dataset...'
wget -nc https://zenodo.org/record/19025/files/sscm_data.tar.xz

# Unzip the dataset.
[ -d sscm_data/ ] && echo 'Already untarred.' || tar -xzf sscm_data.tar.gz


mkdir -p example_data
# Get benign training data.
cat sscm_data/train/benign.raw | shuf | head -n10000 > example_data/benign.raw

# Get simulated training data.
cat sscm_data/train/sim-1.raw | shuf | head -n10000 > example_data/sim.raw

# Get 1000G and clinvar test data.
cat sscm_data/test/1000G-benign.raw | shuf | head -n10000 > example_data/1000G-benign.raw
cat sscm_data/test/clinvar-pathogenic.raw | shuf | head -n10000 > example_data/clinvar-pathogenic.raw

# Train the model using the simulated and benign datasets.
bin/sscm-train first_model --replace --feature_file sscm/data/features.json --train example_data/sim.raw  --fit example_data/benign.raw  --features verPhyloP verPhCons Consequence

# Test the model on subsets of 1000G benign and clinvar
bin/sscm-predict first_model --feature_file sscm/data/features.json < example_data/1000G-benign.raw > example_data/1000G-benign-annotated.raw
bin/sscm-predict first_model --feature_file sscm/data/features.json < example_data/clinvar-pathogenic.raw > example_data/clinvar-pathogenic-annotated.raw
