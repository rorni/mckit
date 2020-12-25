#
# Jupyter kernel setup
#
#
# dvp, Dec 2020
#

mckit=${1:-mckit}

echo "Creating jupyter kernel for conda environment $mckit"
python -m ipykernel install --user --name $mckit
if [[ $? ]]; then
    echo "To use $mckit environment in jupyter:"
    echo "  - Run 'jupyter lab'"
    echo "  - Open or create notebook"
    echo "  - Select kernel $mckit"
    echo "  - check if 'import mckit' in the notebook works"
else
    echo "ERROR: kernel $mckit is not created"
fi