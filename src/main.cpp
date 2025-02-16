/**
 * @file main.cpp
 * @author Tathagata Guha Ray
 * @brief
 * This program is aimed at automatically blurring all the faces found in an image
 * using the NVIDIA Performance Primitives (NPP) library.
 * The program takes in the path to an image file and the path to the output file 
 * as command line arguments.
 * 
 * @version 1.0
 * @date 16-02-2025
 * 
 */

// STL includes
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>

// OpenCV includes
#include <opencv2/opencv.hpp>

// CUDA and NPP includes
#include <npp.h>
#include <cuda_runtime.h>

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include "helper_cuda.h"
#include "helper_string.h"



std::tuple<std::string, std::string> parseCommandLineArgs(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cerr << "Usage: " << argv[0] << " <image_path> <output_path>" << std::endl;
		exit(EXIT_FAILURE);
	}

	return {argv[1], argv[2]};
}


void checkCompatibility()
{
	// Check NPP and CUDA versions
    std::cout << "NPP version: " << nppGetLibVersion() << std::endl;
    std::cout << "CUDA version: " << CUDART_VERSION << std::endl;

	// Check CUDA capabilities
    if (!checkCudaCapabilities(1, 0))
    {
        std::cerr << "[ERROR]: CUDA capability 1.0 is required" << std::endl;
		exit(EXIT_FAILURE);
    }

	// Check OpenCV version
	std::cout << "OpenCV version: " << CV_VERSION << std::endl;
}


int main(int argc, char *argv[])
{
	// First check for correct NPP, CUDA and OpenCV versions
    checkCompatibility();
	
	// Parse command line arguments
    auto [imagePath, outPath] = parseCommandLineArgs(argc, argv);

	// Check if image exists
	std::ifstream file(imagePath.data(), std::ifstream::in);

	if (!file.good())
	{
		std::cerr << "[ERROR]: Image file does not exist" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	


	return 0;
}
