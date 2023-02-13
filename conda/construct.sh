MAGUS_VERSION=$(python -c "import magus; print(magus.__version__)")
export MAGUS_VERSION=$MAGUS_VERSION
zip -r examples.zip ../examples
mv examples.zip construct
cp ../LICENSE construct
constructor construct
