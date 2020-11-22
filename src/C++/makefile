CXX       := nvcc
CXX_FLAGS := -lcublas -lcurand

BIN := bin
SRC := src
INCLUDE := include
EXECUTABLE := somwork_cuda
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
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM.cu -o $(BIN)/$(LIBRARY)$(OBJEXT)
	ar -rcs $(BIN)/$(LIBRARY)$(LIBEXT) $(BIN)/$(LIBRARY)$(OBJEXT)

build: library
	@echo ""
	@echo "Building somwork executable..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM_work.cpp -o $(BIN)/$(EXECUTABLE)$(OBJEXT)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $(BIN)/$(EXECUTABLE)$(OBJEXT) -o $(BIN)/$(EXECUTABLE) $(BIN)/$(LIBRARY)$(LIBEXT)

clean:
	@echo "Clearing..."
	rm $(BIN)/*
