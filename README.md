<h1 align="center">FastBlur CUDA Assignment</h1>

## Purpose
- This project is part of the Coursera CUDA Assignment, specifically, Cuda at Scale for the Enterprise.
- This project aims to demonstrate the use of CUDA for high performance image processing and to provide a hands-on experience with NPP (NVIDIA Performance Primitives).

&nbsp;
- **NOTE for Peer Reviewers**: Originally the idea was to implement a program that would detect human faces in an image and blur all of them using NPP. Due to OpenCV not properly building with CUDA for the longest time, I decided to implement a simple image blur instead.
- As such there are includes for OpenCV in the code, but they are not used. The code is still functional and demonstrates the use of NPP for image processing.
- I'm still hard at work trying to get OpenCV to build with CUDA, and when that happens, I'll drop another project.

## Prerequisites
Before running this project, ensure you have the following installed:
- CUDA Toolkit (12.8 in my case)
- NVIDIA GPU with CUDA capability
- C++ Compiler (e.g., GCC)
- Make

## How to Run
1. Clone the repository:
    ```sh
    git clone https://github.com/AlexJMercer/CUDA-NPP-Assignment.git
    cd CUDA-NPP-Assignment
    ```
2. Compile the code:
    ```sh
    make clean all
    ```
3. Run the executable:
    ```sh
    ./main
    ```

## Final Thoughts
- I have built this project inside WSL2 with Ubuntu 24.04 LTS, and ran on a NVIDIA GTX 1650.
- Learnt a great deal about NPP and CUDA in general, however, proper knowledge comes with building even more projects. 

This is only the beginning.