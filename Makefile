# tool macros
CC := nvcc
CCFLAGS := -std=c++17
CCOBJFLAGS := $(CCFLAGS) -x cu -c -O3
LIBS := -l SDL2

# other variables
EXECUTABLE := main

MKD := mkdir

# path macros
BIN_PATH := bin
OBJ_PATH := obj
SRC_PATH := src
CINC := -I inc

# compile macros
APP := $(BIN_PATH)/$(EXECUTABLE)

# src files & obj files
SRC := $(wildcard $(SRC_PATH)/*.c*)
OBJ := $(subst $(SRC_PATH), $(OBJ_PATH), $(SRC:.cpp=.o))


# Clean files list
CLEAN_LIST := $(OBJ_PATH)/*.o \
	$(BIN_PATH)/* \

# === RULES ===
# default rule
default: makedir all

# Compile app executable by linking objects
$(APP): $(OBJ)
	$(CC) $(CCFLAGS) $(CINC) $(LIBS) -o $@ $(OBJ)

# Compile objects
$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	@$(MKD) -p $(dir $@)
	$(CC) $(CCOBJFLAGS) $(CINC) $(LIBS) -o $@ $<

# === PHONY RULES ===
.PHONY: makedir
makedir:
	@$(MKD) -p $(BIN_PATH) $(OBJ_PATH)

.PHONY: all
all: $(APP)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm $(CLEAN_LIST)
