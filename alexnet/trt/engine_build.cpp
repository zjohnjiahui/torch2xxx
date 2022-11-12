#include "NvInfer.h"
#include "logging.h"
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

using namespace nvinfer1;

// Logger from TRT API
static Logger gLogger;

#define MAX_BATCH_SIZE 2
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

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,
                            ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;  // scale
    float *beta = (float*)weightMap[lname + ".bias"].values;   //  bias
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;

    int len = weightMap[lname + ".running_var"].count;
    std::cout << lname << " BN len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

/**
 * Create Engine using the TRT Builder and Configurations
 *
 * @param maxBatchSize: batch size for built TRT model
 * @param builder: to build engine and networks
 * @param config: configuration related to Hardware
 * @param dt: datatype for model layers
 * @return engine: TRT model
 */
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    std::cout << "[INFO]: Creating MLP using TensorRT..." << std::endl;

    // Load Weights from relevant file
    std::map<std::string, Weights> weightMap = loadWeights("../alexnet.wts");

    // Create an empty network
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create an input with proper *name
    ITensor *data = network->addInput("data", dt, Dims3{3, 224, 224});  // 32x32
    assert(data);

    // conv bn relu maxpool
    IConvolutionLayer* conv1 =  network->addConvolutionNd(*data, 48, DimsHW{11, 11},
                              weightMap["conv1.conv.weight"], weightMap["conv1.conv.bias"]);
    conv1->setStrideNd(DimsHW{4, 4});
    conv1->setPaddingNd(DimsHW{2, 2});
    IScaleLayer* scale1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "conv1.bn", 1e-5);
    IActivationLayer* act1 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* pool1 = network->addPoolingNd(*act1->getOutput(0),PoolingType::kMAX, DimsHW{3,3});
    pool1->setStrideNd(DimsHW{2, 2});

    // conv bn relu maxpool
    IConvolutionLayer* conv2 =  network->addConvolutionNd(*pool1->getOutput(0), 128, DimsHW{5, 5},
                              weightMap["conv2.conv.weight"], weightMap["conv2.conv.bias"]);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{2, 2});
    IScaleLayer* scale2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "conv2.bn", 1e-5);
    IActivationLayer* act2 = network->addActivation(*scale2->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* pool2 = network->addPoolingNd(*act2->getOutput(0),PoolingType::kMAX, DimsHW{3,3});
    pool2->setStrideNd(DimsHW{2, 2});

    // conv bn relu
    IConvolutionLayer* conv3 =  network->addConvolutionNd(*pool2->getOutput(0), 192, DimsHW{5, 5},
                              weightMap["conv3.conv.weight"], weightMap["conv3.conv.bias"]);
    conv3->setStrideNd(DimsHW{1, 1});
    conv3->setPaddingNd(DimsHW{2,2});
    IScaleLayer* scale3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "conv3.bn", 1e-5);
    IActivationLayer* act3 = network->addActivation(*scale3->getOutput(0), ActivationType::kRELU);

    // conv bn relu
    IConvolutionLayer* conv4 =  network->addConvolutionNd(*act3->getOutput(0), 192, DimsHW{5, 5},
                              weightMap["conv4.conv.weight"], weightMap["conv4.conv.bias"]);
    conv4->setStrideNd(DimsHW{1, 1});
    conv4->setPaddingNd(DimsHW{2,2});
    IScaleLayer* scale4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), "conv4.bn", 1e-5);
    IActivationLayer* act4 = network->addActivation(*scale4->getOutput(0), ActivationType::kRELU);

    // conv bn relu maxpool
    IConvolutionLayer* conv5 =  network->addConvolutionNd(*act4->getOutput(0), 128, DimsHW{5, 5},
                              weightMap["conv5.conv.weight"], weightMap["conv5.conv.bias"]);
    conv5->setStrideNd(DimsHW{1, 1});
    conv5->setPaddingNd(DimsHW{2, 2});
    IScaleLayer* scale5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), "conv5.bn", 1e-5);
    IActivationLayer* act5 = network->addActivation(*scale5->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* pool5 = network->addPoolingNd(*act5->getOutput(0),PoolingType::kMAX, DimsHW{3,3});
    pool5->setStrideNd(DimsHW{2, 2});

    // fc relu
    IFullyConnectedLayer *fc1 = network->addFullyConnected(*pool5->getOutput(0), 2048,
                                                               weightMap["classifier.fc1.weight"],
                                                               weightMap["classifier.fc1.bias"]);
    IActivationLayer* relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);

    //fc relu
    IFullyConnectedLayer *fc2 = network->addFullyConnected(*relu1->getOutput(0), 2048,
                                                               weightMap["classifier.fc2.weight"],
                                                               weightMap["classifier.fc2.bias"]);
    IActivationLayer* relu2 = network->addActivation(*fc2->getOutput(0), ActivationType::kRELU);

    // fc softmax
    IFullyConnectedLayer *fc3 = network->addFullyConnected(*relu2->getOutput(0), 5,
                                                               weightMap["classifier.fc3.weight"],
                                                               weightMap["classifier.fc3.bias"]);
    ISoftMaxLayer* soft = network->addSoftMax(*fc3->getOutput(0));
    soft->setAxes(1<<0);
    Dims a = soft->getOutput(0)->getDimensions();
    std::cout <<a.d[0] <<" "<< a.d[1] <<" "<< a.d[2] << std::endl;

    soft->getOutput(0)->setName("out");
    network->markOutput(*soft->getOutput(0));

    // Set configurations
    builder->setMaxBatchSize(maxBatchSize);
    // Set workspace size
    config->setMaxWorkspaceSize(1 << 28); // 256M
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
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
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
    std::ofstream p("../alexnet.engine", std::ios::binary);
    if (!p) {
        std::cerr << "[ERROR]: could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    // Release the memory
    modelStream->destroy();

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\tRun inference using `./alexnet -d`" << std::endl;

}

