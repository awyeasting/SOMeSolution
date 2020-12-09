CUDA         := /usr/local/cuda-11.1

NVCC         := nvcc
MPICXX 	     := mpicxx
MPICXX_FLAGS := -I$(CUDA)/include
NVCC_FLAGS   := -lcublas -lcurand
CXX          := g++
CXX_FLAGS    :=

MPI := $(HOME)/opt/openmpi
BIN := bin
SRC := src
INCLUDE := include
EXECUTABLE := somwork_mpicuda
LIBRARY := somesolution

ifeq ($(OS),Windows_NT)
	OBJEXT := .obj
	LIBEXT := .lib
else
	OBJEXT := .o
	LIBEXT := .a
endif

all: library

run: clean build
	clear
	@echo ""
	@echo "Executing..."
	./$(BIN)/$(EXECUTABLE)

library:
	@echo "Building somesolution library..."
	$(MPICXX) $(MPICXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM.cpp -o $(BIN)/$(LIBRARY)_mpi$(OBJEXT)
	$(NVCC) $(NVCC_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM.cu -o $(BIN)/$(LIBRARY)_cuda$(OBJEXT)
	ar -crs $(BIN)/$(LIBRARY)$(LIBEXT) $(BIN)/$(LIBRARY)_cuda$(OBJEXT) $(BIN)/$(LIBRARY)_mpi$(OBJEXT)

build: library
	@echo ""
	@echo "Building somwork executable..."
	$(MPICXX) $(MPICXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM_work.cpp -o $(BIN)/$(EXECUTABLE)$(OBJEXT)
	mpicxx bin/somesolution_mpi.o bin/somesolution_cuda.o bin/somwork_mpicuda.o -Iinclude -lcublas -lcurand -lcudart -L/usr/local/cuda-11.1/lib64 -o bin/somwork_mpicuda -fopenmp

extra:
	$(MPICXX) $(MPICXX_FLAGS) -I$(INCLUDE) $(NVCC_FLAGS) $(BIN)/$(EXECUTABLE)$(OBJEXT) $(BIN)/$(LIBRARY)$(LIBEXT) -o $(BIN)/$(EXECUTABLE) -fopenmp

clean:
	@echo "Clearing..."
	rm $(BIN)/*
