set -uex
mkdir -p .cache .images
SIM_RES_PATH=".cache/$1_$2.res"
IMAGE_PATH=".images/$1_$2.png"
./venv/bin/python3 simulation_main.py --simulation-name $1 --render-to ${IMAGE_PATH} --load-from ${SIM_RES_PATH} && open ${IMAGE_PATH}