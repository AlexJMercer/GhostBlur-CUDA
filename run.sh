#!/bin/bash

# Clean and build the project
make clean all

# Run the project with the given arguments
# Arguments: $1 - input file, $2 - output file
./bin/main "$1" "$2"
