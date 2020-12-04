CXX       := nvcc
CXX_FLAGS := -lcublas -lcurand -Xcompiler " --openmp"

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
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) -I$(MPI)/include -L$(MPI)/lib -lmpi $(SRC)/SOM.cu -o $(BIN)/$(LIBRARY)$(OBJEXT)
	ar -rcs $(BIN)/$(LIBRARY)$(LIBEXT) $(BIN)/$(LIBRARY)$(OBJEXT)

build: library
	@echo ""
	@echo "Building somwork executable..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) -I$(MPI)/include -L$(MPI)/lib -lmpi $(SRC)/SOM_work.cpp -o $(BIN)/$(EXECUTABLE)$(OBJEXT)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -I$(MPI)/include -L$(MPI)/lib -lmpi $(BIN)/$(EXECUTABLE)$(OBJEXT) -o $(BIN)/$(EXECUTABLE) $(BIN)/$(LIBRARY)$(LIBEXT)

clean:
	@echo "Clearing..."
	rm $(BIN)/*
