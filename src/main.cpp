#include <iostream>
#include <string>
#include <vector>
#include <tuple>

#include <npp.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_string.h"



std::tuple<std::string, std::string> parseCommandLineArgs(int argc, char *argv[])
{
    return {};  // Return empty vector for now
}


void checkCompatibility()
{
    std::cout << "NPP version: " << nppGetLibVersion() << std::endl;
    std::cout << "CUDA version: " << CUDART_VERSION << std::endl;

    if (!checkCudaCapabilities(1, 0))
    {
        std::cerr << "[ERROR]: CUDA capability 1.0 is required" << std::endl;
		exit(EXIT_FAILURE);
    }
}


int main(int argc, char *argv[])
{
    auto [imagePath, outPath] = parseCommandLineArgs(argc, argv);

    // First check for correct NPP and CUDA versions
    checkCompatibility();
}
