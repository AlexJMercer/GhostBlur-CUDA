# Created by: Tathagata Guha Ray
# Date: 16th February 2025

# Windows-specific settings
OSUPPER := WINDOWS
OSLOWER := windows

# Directories
SRCDIR := src
OBJDIR := obj
BINDIR := bin
LIBDIR := lib
DATADIR := data

# File extensions
SRCEXT := cpp
OBJEXT := obj
EXEEXT := .exe

# CUDA settings
CUDA_PATH ?= "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
NVCC := $(CUDA_PATH)\bin\nvcc.exe

# Compiler flags
CCFLAGS := -gencode arch=compute_52,code=sm_52 -O2 -m64 -I$(LIBDIR) -I$(LIBDIR)/UtilNPP
LDFLAGS := -L"$(CUDA_PATH)\lib\x64" -lcudart -lnppc -lnppial

# Files
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%.$(SRCEXT), $(OBJDIR)/%.$(OBJEXT), $(SOURCES))
TARGET := $(BINDIR)/main$(EXEEXT)

# Default rule
all: $(TARGET)

# Ensure obj directory exists before compiling
$(OBJDIR):
	mkdir $(OBJDIR)

# Compile each .cpp file separately
$(OBJDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT) | $(OBJDIR)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

# Link object files into the final executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

# Clean rule
clean:
	del /F /Q /S $(OBJDIR)\*.$(OBJEXT) $(BINDIR)\*$(EXEEXT)
	