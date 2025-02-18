# Created by: Tathagata Guha Ray
# Date: 16th February 2025
# OS-specific settings
OSUPPER := LINUX
OSLOWER := linux
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
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
LIBS = -lnppc -lnppial -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lcudart
# OpenCV settings
OPENCV_PATH ?= /usr/local
OPENCV_INCLUDE := $(OPENCV_PATH)/include/opencv4
OPENCV_LIB := $(OPENCV_PATH)/lib
OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudaobjdetect
# FreeImage settings
FREEIMAGE_PATH ?= /usr/local
FREEIMAGE_INCLUDE := $(FREEIMAGE_PATH)/include
FREEIMAGE_LIB := $(FREEIMAGE_PATH)/lib
FREEIMAGE_LIBS := -lfreeimage
# Compiler flags
CCFLAGS := -gencode arch=compute_52,code=sm_52 -O2 -m64 -std=c++17 -Wno-deprecated-gpu-targets -I$(LIBDIR) -I$(LIBDIR)/UtilNPP -I$(OPENCV_INCLUDE) -I$(FREEIMAGE_INCLUDE) 
LDFLAGS := -L$(CUDA_PATH)/lib64 -L/usr/lib/x86_64-linux-gnu -lcudart $(LIBS) -L$(OPENCV_LIB) $(OPENCV_LIBS) -L$(FREEIMAGE_LIB) $(FREEIMAGE_LIBS)
# Files
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%.$(SRCEXT), $(OBJDIR)/%.$(OBJEXT), $(SOURCES))
TARGET := $(BINDIR)/main$(EXEEXT)
# Default rule
all: $(TARGET)
# Ensure obj directory exists before compiling
$(OBJDIR):
	mkdir -p $(OBJDIR)
# Compile each .cpp file separately
$(OBJDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT) | $(OBJDIR)
	$(NVCC) $(CCFLAGS) -c -o $@ $<
# Link object files into the final executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)
# Clean rule
clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/main