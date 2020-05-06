CXX       := g++
CXX_FLAGS := -fopenmp -std=c++17

BIN := bin
SRC := src
INCLUDE := include
EXECUTABLE := somwork
LIBRARY := somesolution

all: library

run: clean build
	clear
	@echo ""
	@echo "Executing..."
	./$(BIN)/$(EXECUTABLE)

library:
	@echo "Building somesolution library..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM.cpp -o $(BIN)/$(LIBRARY).o
	ar rcs $(BIN)/$(LIBRARY).a $(BIN)/$(LIBRARY).o

build: library
	@echo ""
	@echo "Building somwork executable..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM_work.cpp -o $(BIN)/$(EXECUTABLE).o
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $(BIN)/$(EXECUTABLE).o -o $(BIN)/$(EXECUTABLE) $(BIN)/$(LIBRARY).a 

clean:
	@echo "Clearing..."
	rm $(BIN)/*
