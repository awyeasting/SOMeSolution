CXX       := g++
CXX_FLAGS := -fopenmp -std=c++17

BIN := bin
SRC := src
INCLUDE := include
EXECUTABLE := somesolution
OBJECT := somesolution.o
LIBRARY := somesolution.a

all: library

run: clean build
	clear
	@echo "Executing..."
	./$(BIN)/$(EXECUTABLE)

library:
	@echo "Building SOMeSolution library..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM.cpp -o $(BIN)/$(OBJECT)
	ar rcs $(BIN)/$(LIBRARY) $(BIN)/$(OBJECT)

build: library
	@echo "Building SOMeSolution executable..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM_work.cpp -o $(BIN)/b$(OBJECT)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $(BIN)/b$(OBJECT) -o $(BIN)/$(EXECUTABLE) $(BIN)/$(LIBRARY) 

clean:
	@echo "Clearing..."
	-rm $(BIN)/*
