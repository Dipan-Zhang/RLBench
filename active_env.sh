#!/bin/bash
# activate the environment for running RLBench using: source active_env.sh
export COPPELIASIM_ROOT="${HOME}/software/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"