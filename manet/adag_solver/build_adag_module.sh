g++ -O3 -Wall -shared -std=c++11 -fPIC libadag.cpp -o libadag.so
python3 build_adag_cffi.py
