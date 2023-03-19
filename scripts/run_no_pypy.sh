set -uex
mkdir -p .cache .images
ESCAPED_SIM_NAME=${1//\//_}
SIM_RES_PATH=".cache/${ESCAPED_SIM_NAME}_$2.res"
IMAGE_PATH=".images/${ESCAPED_SIM_NAME}_$2.png"

# Save dump to be able to visualize it w/o running simulation.
./venv/bin/python3 simulation_main.py --simulation-name $1 --target-rps $2 --save-to ${SIM_RES_PATH} --render-to ${IMAGE_PATH} && open ${IMAGE_PATH}