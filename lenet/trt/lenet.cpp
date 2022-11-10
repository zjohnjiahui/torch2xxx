#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "logging.h"


using namespace nvinfer1;


void performSerialization(); // build engine from wts
void performInference(); // run trt engine

/**
 * Parse command line arguments
 *
 * @param argc: argument count
 * @param argv: arguments vector
 * @return int: a flag to perform operation
 */
int checkArgs(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "[ERROR]: Arguments not right!" << std::endl;
        std::cerr << "./lenet -s   // serialize model to engine file" << std::endl;
        std::cerr << "./lenet -d   // deserialize engine file and run inference" << std::endl;
        return -1;
    }
    if (std::string(argv[1]) == "-s") {
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        return 2;
    }
    return -1;
}

int main(int argc, char** argv)
{
    int args = checkArgs(argc, argv);
    if (args == 1){
        printf("[INFO]: Serialization\n");
        performSerialization();
    }
    else if (args == 2){
        printf("[INFO]: Inference\n");
        performInference();
    }
    return 0;
}
