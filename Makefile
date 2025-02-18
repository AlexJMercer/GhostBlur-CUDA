# Created by: Tathagata Guha Ray
# Date: 16th February 2025

# Directories
SRCDIR := src
OBJDIR := obj
BINDIR := bin
LIBDIR := lib
DATADIR := data

# File extensions
SRCEXT := cpp
OBJEXT := o
EXEEXT :=

# CUDA settings
CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# Libraries
LIBS := -lnppc -lnppidei -lnppial -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lcudart

# Compiler flags
NVCCFLAGS := -gencode arch=compute_75,code=sm_75 -O3 -m64 -std=c++17 -I$(LIBDIR) -I$(LIBDIR)/UtilNPP -I/usr/include/stb

# Linking flags
LDFLAGS := -L$(CUDA_PATH)/lib64 $(LIBS) -L/usr/lib -lfreeimage

# Files
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%.$(SRCEXT), $(OBJDIR)/%.$(OBJEXT), $(SOURCES))
TARGET := $(BINDIR)/main$(EXEEXT)

# Default rule
all: $(TARGET)

# Ensure obj directory exists before compiling
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile each .cpp file separately (host code with nvcc)
$(OBJDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Link object files into the final executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Clean rule
clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/main
