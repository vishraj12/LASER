module purge
module load StdEnv/2020
module load python/3.8.10
export LASER_ROOT=$HOME/scratch/LASER
source $LASER_ROOT/.venv/bin/activate
export PYTHONPATH=$LASER_ROOT/PythonAPI/carla:$LASER_ROOT/scenario_runner:$LASER_ROOT/leaderboard:$PYTHONPATH
