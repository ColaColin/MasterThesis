#! /bin/bash

# Run without parameters to build
# Run with a single parameter "clean" to clean generated files
# Run with a single parameter "test" to run tests
# Verbose tests can also be useful: "testverbose"

if [[ $# -eq 0 ]] ; then
    python setup.py build_ext --inplace
elif [ $1 == "clean" ]; then
    clear;
    python setup.py clean
    echo Clean complete. You may build now by passing in no arguments.
elif [ $1 == "test" ]; then
    python -m unittest
elif [ $1 == "testverbose" ]; then
    python -m unittest -v
elif [ $1 == "testworker" ]; then
    python -m unittest core.testcases.testBson
    python -m unittest core.testcases.testConnect4
    python -m unittest core.testcases.testMCTS0
    python -m unittest core.testcases.testMNK
    python -m unittest core.testcases.testPytorchPolicy
    python -m unittest core.testcases.testTempMoves
    python -m unittest core.testcases.testTestGame
fi