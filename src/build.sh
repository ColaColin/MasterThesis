if [[ $# -eq 0 ]] ; then
    python setup.py build_ext --inplace
elif [ $1 == "clean" ]; then
    clear;
    python setup.py clean
    echo Clean complete. You may build now by passing in no arguments.
fi

