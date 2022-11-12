#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

// Logger from TRT API
static Logger gLogger;

const int INPUT_H = 224;
const int INPUT_W = 224;
const int INPUT_C = 3;
const int INPUT_SIZE = INPUT_C*INPUT_H*INPUT_W;
const int OUTPUT_SIZE = 5; // output data length in an inference
const int BATCH_SIZE = 2;


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
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // execute inference using context provided by engine
    context.enqueue(batchSize, buffers, stream, nullptr);

    // copy output back from device (GPU) to host (CPU)
    cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

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
    std::ifstream file("../alexnet.engine", std::ios::binary);
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

    float out[OUTPUT_SIZE* BATCH_SIZE];  // array for output
    float data[INPUT_SIZE * BATCH_SIZE]; // array for input

    cv::Mat img0 = cv::imread("../0.jpg", cv::IMREAD_COLOR);
    cv::Mat img0_re;
    cv::resize(img0, img0_re,cv::Size(224,224));
    uchar* Img0Data = img0_re.data;

    cv::Mat img1 = cv::imread("../1.jpg", cv::IMREAD_COLOR);
    cv::Mat img1_re;
    cv::resize(img1, img1_re,cv::Size(224,224));
    uchar* Img1Data = img1_re.data;

    uchar* imgData[2] = {Img0Data, Img1Data};

    // bgrbgrbgr -> rrrgggbbb
    for(int batch=0;batch<BATCH_SIZE;++batch)
    {
        for(int i=0;i<INPUT_H*INPUT_W;++i)
        {
            data[batch*INPUT_SIZE + i] = (imgData[batch][i*3+2]/255.0 - 0.485)/0.229;
            data[batch*INPUT_SIZE + i + INPUT_H * INPUT_W]= (imgData[batch][i*3+1]/255.0 - 0.456)/0.224;
            data[batch*INPUT_SIZE + i + 2* INPUT_H * INPUT_W]= (imgData[batch][i*3]/255.0 - 0.406)/0.225;
        }
    }


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

    std::cout << "\n[INFO]: Output:\t";
    for(int batch=0;batch<BATCH_SIZE;++batch)
    {
        int max_index = 0;
        float max = 0;
        for(int i=0;i<OUTPUT_SIZE;++i)
        {
            if(out[batch* OUTPUT_SIZE+ i] > max){
                max = out[batch* OUTPUT_SIZE+ i];
                max_index = i;
            }

            std::cout << out[batch* OUTPUT_SIZE+ i] << " ";
        }
       // std::cout << max_index  << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
