LEVEL=4
#python3 sgpp_precalcSparseGridPoints.py --test_strategy='individual-testing' --level=$LEVEL;
python3 sgpp_precalcSparseGridPoints.py --test_strategy='two-stage-testing' --level=$LEVEL;
#python3 sgpp_precalcSparseGridPoints.py --test_strategy='binary-splitting' --level=$LEVEL;
#python3 sgpp_precalcSparseGridPoints.py --test_strategy='RBS' --level=$LEVEL;
#python3 sgpp_precalcSparseGridPoints.py --test_strategy='purim' --level=$LEVEL;
#python3 sgpp_precalcSparseGridPoints.py --test_strategy='sobel' --level=$LEVEL;