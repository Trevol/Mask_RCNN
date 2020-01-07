REL_TO_PROJ_ROOT="../../.." #to Mask_RCNN
CURRENT_DIR=$(pwd)

cd $REL_TO_PROJ_ROOT

PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH

source venv/bin/activate

cd $CURRENT_DIR
