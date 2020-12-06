CXX       := g++
CXX_FLAGS :=

BIN := bin
SRC := src
INCLUDE := include
EXECUTABLE := somwork
LIBRARY := somesolution

ifeq ($(OS),Windows_NT)
	OBJEXT := .obj 
	LIBEXT := .lib
else
	OBJEXT := .o 
	OBJEXT := .a
endif

all: library

library:
	@echo "Building somesolution library..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM.cpp -o $(BIN)/$(LIBRARY)$(OBJEXT)
	ar -rcs $(BIN)/$(LIBRARY)$(LIBEXT) $(BIN)/$(LIBRARY)$(OBJEXT)

build: library
	@echo ""
	@echo "Building somwork executable..."
	$(CXX) $(CXX_FLAGS) -c -I$(INCLUDE) $(SRC)/SOM_work.cpp -o $(BIN)/$(EXECUTABLE)$(OBJEXT)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $(BIN)/$(EXECUTABLE)$(OBJEXT) -o $(BIN)/$(EXECUTABLE) $(BIN)/$(LIBRARY)$(LIBEXT)

clean:
	@echo "Clearing..."
	rm $(BIN)/*