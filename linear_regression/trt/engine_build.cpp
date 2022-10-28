#include "NvInfer.h"
#include "logging.h"
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace nvinfer1;

// Logger from TRT API
static Logger gLogger;

#define MAX_BATCH_SIZE 4
#define USE_FP16 1

/**
 * Parse the .wts file and store weights in dict format.
 *
 * @param file path to .wts file
 * @return weight_map: dictionary containing weights and their values
 */
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "[INFO]: Loading weights..." << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open Weight file
    std::ifstream input(file);
    assert(input.is_open() && "[ERROR]: Unable to load weight file...");

    // Read number of weights
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // wts file format : weight_name weight_size weight0 weight1 ...

    // Loop through number of line, actually the number of weights & biases
    while (count--) {
        // TensorRT weights
        Weights wt{DataType::kFLOAT, nullptr, 0}; // { type arr_pointer arr_size }
        uint32_t size;
        // Read name and type of weights
        std::string w_name;
        input >> w_name >> std::dec >> size;  // decimal number
        wt.type = DataType::kFLOAT;

        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0 ; x < size; ++x) {
            // Change hex values to uint32 (for higher values)
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;

        // Add weight values against its name (key)
        weightMap[w_name] = wt;
    }
    return weightMap;
}

/**
 * Create Linear Perceptron using the TRT Builder and Configurations
 *
 * @param maxBatchSize: batch size for built TRT model
 * @param builder: to build engine and networks
 * @param config: configuration related to Hardware
 * @param dt: datatype for model layers
 * @return engine: TRT model
 */
ICudaEngine *createLinearEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    std::cout << "[INFO]: Creating Linear using TensorRT..." << std::endl;

    // Load Weights from relevant file
    std::map<std::string, Weights> weightMap = loadWeights("../../linear_regression.wts");

    // Create an empty network
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create an input with proper *name
    ITensor *data = network->addInput("data", dt, Dims3{1, 1, 1});
    assert(data);

    // Add layer for Linear
    IFullyConnectedLayer *fc1 = network->addFullyConnected(*data, 1,
                                                           weightMap["linear.weight"],
                                                           weightMap["linear.bias"]);
    assert(fc1);

    // set output with *name
    fc1->getOutput(0)->setName("out");

    // mark the output
    network->markOutput(*fc1->getOutput(0));

    // Set configurations
    builder->setMaxBatchSize(maxBatchSize);
    // Set workspace size
    config->setMaxWorkspaceSize(1 << 20); // 1M
    // Set inference type
#if USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif

    // Build CUDA Engine using network and configurations
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    // Don't need the network any more
    // free captured memory
    network->destroy();

    // Release host memory
    for (auto &mem: weightMap) {
        free((void *) (mem.second.values));
    }

    return engine;
}


/**
 * Create engine using TensorRT APIs
 *
 * @param maxBatchSize: for the deployed model configs
 * @param modelStream: shared memory to store serialized model
 */
void APIToBuildModel(unsigned int maxBatchSize, IHostMemory **modelStream) {
    // Create builder with the help of logger
    IBuilder *builder = createInferBuilder(gLogger);

    // Create hardware configs
    IBuilderConfig *config = builder->createBuilderConfig();

    // Build an engine
    ICudaEngine *engine = createLinearEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
    (*modelStream) = engine->serialize();

    // free up the memory
    engine->destroy();
    builder->destroy();
}


/**
 * Serialization Function
 */
void performSerialization() {
    // Shared memory object
    IHostMemory *modelStream{nullptr};

    // Write model into stream
    APIToBuildModel(MAX_BATCH_SIZE, &modelStream);
    assert(modelStream != nullptr);

    std::cout << "[INFO]: Writing engine into binary..." << std::endl;

    // Open the file and write the contents there in binary format
    std::ofstream p("../linear.engine", std::ios::binary);
    if (!p) {
        std::cerr << "[ERROR]: could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    // Release the memory
    modelStream->destroy();

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\tRun inference using `./linear -d`" << std::endl;

}

