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

// CUDA and NPP includes
#include <npp.h>
#include <cuda_runtime.h>

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
}



void blurFacesNPP(const std::string& imagePath, const std::string& outputPath)
{
	int width, height, channels;
    unsigned char* img = stbi_load(imagePath.c_str(), &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "[ERROR] Could not read image!" << std::endl;
        return;
    }

    // Convert image to GPU memory
    Npp8u *d_src;
    int imgSize = width * height * 3;
    cudaMalloc((void **)&d_src, imgSize);
    cudaMemcpy(d_src, img, imgSize, cudaMemcpyHostToDevice);

    // Dummy face detection (replace with actual implementation)
    std::vector<NppiRect> faces = {{50, 50, 100, 100}}; // Example face region

    // Apply NPP Gaussian Blur to detected faces
    for (const auto &face : faces) 
    {
        NppiSize roiSize = {face.width, face.height};
        int stepSize;
        Npp8u *d_faceRoi = nppiMalloc_8u_C3(face.width, face.height, &stepSize);
        nppiCopy_8u_C3R(d_src + face.y * width * 3 + face.x * 3, width * 3, d_faceRoi, stepSize, roiSize);
        nppiFilterGauss_8u_C3R(d_faceRoi, stepSize, d_faceRoi, stepSize, roiSize, NPP_MASK_SIZE_5_X_5);
        nppiCopy_8u_C3R(d_faceRoi, stepSize, d_src + face.y * width * 3 + face.x * 3, width * 3, roiSize);
        nppiFree(d_faceRoi);
    }

    // Copy processed image back to host
    cudaMemcpy(img, d_src, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_src);

    // Save output image
    stbi_write_jpg(outputPath.c_str(), width, height, 3, img, 100);
    stbi_image_free(img);
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
	
	blurFacesNPP(imagePath, outPath);

	return 0;
}
