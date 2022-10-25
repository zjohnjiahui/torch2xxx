#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>

using namespace nvinfer1;

// Logger from TRT API
static Logger gLogger;

const int INPUT_SIZE = 1;
const int OUTPUT_SIZE = 1;
const int BATCH_SIZE = 4;


/**
 * Perform inference using the CUDA context
 *
 * @param context: context created by engine
 * @param input: input from the host
 * @param output: output to save on host
 * @param batchSize: batch size for TRT model, it shall be smaller than the maxBatchSize of the engine file
 */
void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    // Get engine from the context
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("data");
    const int outputIndex = engine.getBindingIndex("out");

    // Create GPU buffers on device -- allocate memory for input and output
    cudaMalloc(&buffers[inputIndex], batchSize * INPUT_SIZE * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float));

    // create CUDA stream for simultaneous CUDA operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // copy input from host (CPU) to device (GPU)  in stream
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);

    // execute inference using context provided by engine
    context.enqueue(batchSize, buffers, stream, nullptr);

    // copy output back from device (GPU) to host (CPU)
    cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);

    // synchronize the stream to prevent issues
    //      (block CUDA and wait for CUDA operations to be completed)
    cudaStreamSynchronize(stream);

    // Release stream and buffers (memory)
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}


/**
 * Get inference using the generated engine file
 */
void performInference() {
    // stream to write model
    char *trtModelStream{nullptr};
    size_t size{0};

    // read model from the engine file
    std::ifstream file("../linear.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);// move pointer to the end of the file
        size = file.tellg(); // get pos of the pointer (file size)
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // create a runtime (required for deserialization of model) with NVIDIA's logger
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // deserialize engine for using the char-stream
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    // create execution context -- required for inference executions
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    float out[BATCH_SIZE];  // array for output
    float data[BATCH_SIZE]; // array for input

    data[0] = 4.0; // put any value for input  4*2+3=15
    data[1] = 5.0;
    data[2] = 6.0;
    data[3] = 7.0;

    // time the execution
    auto start = std::chrono::system_clock::now();

    // do inference using the parameters
    doInference(*context, data, out, BATCH_SIZE);

    // time the execution
    auto end = std::chrono::system_clock::now();
    std::cout << "\n[INFO]: Time taken by execution: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


    // free the captured space
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::cout << "\nInput:\t";
    for (float i: data) {
        std::cout << i << ' ';
    }
    std::cout << "\nOutput:\t";
    for (float i: out) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
}
