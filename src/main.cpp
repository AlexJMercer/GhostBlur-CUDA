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
#include <vector>
#include <string>
#include <tuple>

// OpenCV Includes
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

// CUDA and NPP includes
#include <npp.h>
#include <nppi.h>
#include <cuda_runtime.h>

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

/*
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
*/

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

    if (!checkCudaCapabilities(1,0))
    {
        std::cerr << "[ERROR]: CUDA capability 1.0 is required" << std::endl;
		exit(EXIT_FAILURE);
    }

    // Print device information
    cv::cuda::printCudaDeviceInfo(0);

	// Check OpenCV version
	std::cout << "OpenCV version: " << CV_VERSION << std::endl;
}



void blurImage(const std::string &inputImagePath, const std::string &outputImagePath)
{
    cv::Mat img = cv::imread(inputImagePath, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "[ERROR]: Cannot load image!" << std::endl;
        return;
    }

    int width = img.cols, height = img.rows, step = width*3;

    // Allocate memory in GPU
    Npp8u *d_src, *d_dst;
    d_src = nppiMalloc_8u_C3(width, height, &step);
    d_dst = nppiMalloc_8u_C3(width, height, &step);

    // Copy image to GPU
    cudaError_t err = cudaMemcpy2D(d_src, step, img.data, img.step, width * 3, height, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); 
    }

    // Apply the Gaussian Blur
    NppiSize size = {width, height};
    NppiMaskSize kernel = NPP_MASK_SIZE_9_X_9; // 5x5 Gaussian kernel
    Npp32s anchor = 4;
    nppiFilterGauss_8u_C3R(d_src, step, d_dst, step, size, kernel);

    // Adjust Brightness and Contrast
    // Npp32f contrast = 1.3f;
    // Npp32f brightness = 20;
    // nppiMulC_8u_C3RSfs(d_dst, step, (Npp32f*) &contrast, d_dst, step, size, 0);
    // nppiAddC_8u_C3RSfs(d_dst, step, brightness, d_dst, step, size, 0);

    cv::Mat outputImg(height, width, CV_8UC3);
    
    err = cudaMemcpy2D(outputImg.data, outputImg.step, d_dst, step, width * 3, height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); 
    }

    // Save output image
    cv::imwrite(outputImagePath, outputImg);
    
    // Free GPU memory
    nppiFree(d_src);
    nppiFree(d_dst);
}



int main(int argc, char *argv[])
{
	// Parse command line arguments
    auto [imagePath, outPath] = parseCommandLineArgs(argc, argv);
	
	// First check for correct NPP, CUDA and OpenCV versions
    checkCompatibility();

	// Check if image exists
	std::ifstream file(imagePath.data(), std::ifstream::in);

	if (!file.good())
	{
		std::cerr << "[ERROR]: Image file does not exist" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	blurImage(imagePath, outPath);

	return 0;
}
