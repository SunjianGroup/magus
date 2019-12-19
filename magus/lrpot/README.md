``` shell
g++ -std=c++11 -I/fs00/software/anaconda/3-5.0.1/include -I/fs00/software/anaconda/3-5.0.1/include/python3.6m -L/fs00/software/anaconda/3-5.0.1/lib -lboost_python -lboost_numpy -lpython3.6m lrpot.cpp -o lrpot.so -shared -fPIC
```