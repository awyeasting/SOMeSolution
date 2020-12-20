CXX       := g++
CXX_FLAGS := -fopenmp -std=c++17
MPICXX := mpicxx
MPICXX_FLAGS := 

BIN := bin
SRC := src
INCLUDE := include
EXECUTABLE := somwork_mpi
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
	ar rcs $(BIN)/$(LIBRARY)$(LIBEXT) $(BIN)/$(LIBRARY)_mpi$(OBJEXT)

build: library
	@echo ""
	@echo "Building somwork executable..."
	$(MPICXX) $(MPICXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM_work.cpp -o $(BIN)/$(EXECUTABLE)$(OBJEXT)
	$(MPICXX) $(BIN)/$(LIBRARY)_mpi$(OBJEXT) $(BIN)/$(EXECUTABLE)$(OBJEXT) -I$(INCLUDE) -o $(BIN)/$(EXECUTABLE)

clean:
	@echo "Clearing..."
	rm $(BIN)/*