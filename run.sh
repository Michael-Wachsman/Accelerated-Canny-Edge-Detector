#!/bin/bash

# Set paths for input files here
RUN_TYPE="serial"
TEST_IMG="BigDic.jpg"

# define files
INPUT_FILE="images/${TEST_IMG}"
OUTPUT_FILE="images/edges_${TEST_IMG}"
EXECUTABLE="build/${RUN_TYPE}"

# Compile the program
echo "-------Compiling program-------"
rm build/$RUN_TYPE
make $RUN_TYPE

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Execute the compiled code
echo "-------Executing program--------"
$EXECUTABLE --input $INPUT_FILE --output $OUTPUT_FILE

# Check if execution was successful
if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

# # Run correctness check with Python
# echo "-------Checking output correctness-------"
# python3 ./utils/correctness.py $OUTPUT_FILE $REFERENCE_FILE

# if [ $? -ne 0 ]; then
#     echo "Correctness check failed."
#     exit 1
# fi
