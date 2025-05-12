#!/bin/bash

MODELS_URL="https://amsuni-my.sharepoint.com/personal/p_m_ferreira_uva_nl/_layouts/15/download.aspx?share=EdmpWZqOoPhOg9kHa5i0df0B5mUINTHrbeOOOUMS2EJr3Q"
CHECKPOINTS_DIR="checkpoints"

# Check if the checkpoints directory already exists
if [ -d "${CHECKPOINTS_DIR}" ]; then
    echo "Error: '${CHECKPOINTS_DIR}' directory already exists. Aborting."
    exit 1
fi

echo "Downloading checkpoints..."
wget -O checkpoints.tar ${MODELS_URL}

echo "Extracting models..."
tar -xvf checkpoints.tar

rm checkpoints.tar

echo "Done"
