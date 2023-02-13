MAGUS_VERSION=$(python -c "import magus; print(magus.__version__)")
export MAGUS_VERSION=$MAGUS_VERSION
conda build build-recipe --numpy 1.21 -c conda-forge
