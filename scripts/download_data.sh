#!/usr/bin/env bash
# Download the datasets used in the paper.
#
# BRISC 2025 requires a Kaggle API token (~/.kaggle/kaggle.json). See
# https://www.kaggle.com/docs/api for how to obtain one.
#
# Usage:
#   bash scripts/download_data.sh
#
# Paths in configs/*.yaml should then be pointed at the extracted folders.

set -e

mkdir -p data

# --- BRISC 2025 (brain-MRI tumours) ---
echo "Downloading BRISC 2025 ..."
kaggle datasets download -d briscdataset/brisc2025
unzip -q brisc2025.zip -d data/BRISC
rm brisc2025.zip
echo "BRISC 2025 extracted to data/BRISC"

# --- PASCAL VOC 2012 ---
# The official Oxford mirror is used first; if it is slow or down, the
# alternative mirror below serves the same file.
echo "Downloading PASCAL VOC 2012 ..."
VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_MIRROR="https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar"
wget -q "$VOC_URL" -O VOCtrainval_11-May-2012.tar \
    || wget -q "$VOC_MIRROR" -O VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar -C data/
rm VOCtrainval_11-May-2012.tar
echo "PASCAL VOC 2012 extracted to data/VOCdevkit"

echo "Done. Edit configs/*.yaml to point at data/BRISC and data/VOCdevkit."
