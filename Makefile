# Place this file in the repository root (beside starter/ and data/).
CXX ?= g++
CC ?= gcc
CILK_CXX ?= /opt/opencilk-3.0.0/bin/clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Istarter
LDLIBS = -lmatio -pthread

.PHONY: all seq pthreads omp cilk clean
all: seq pthreads omp

build:
	mkdir -p build

build/data.o: starter/data.c starter/data.h | build
	$(CC) -O3 -c starter/data.c -o $@

seq: build/data.o
	$(CXX) $(CXXFLAGS) src/main.cpp build/data.o $(LDLIBS) -o build/tsne_seq

pthreads: build/data.o
	$(CXX) $(CXXFLAGS) -DMODE_PTHREADS src/main.cpp build/data.o $(LDLIBS) -o build/tsne_pthreads

omp: build/data.o
	$(CXX) $(CXXFLAGS) -DMODE_OMP -fopenmp src/main.cpp build/data.o $(LDLIBS) -fopenmp -o build/tsne_omp

cilk: build/data.o
	$(CILK_CXX) $(CXXFLAGS) -DMODE_CILK -fopencilk src/main.cpp build/data.o $(LDLIBS) -fopencilk -o build/tsne_cilk

clean:
	rm -rf build
