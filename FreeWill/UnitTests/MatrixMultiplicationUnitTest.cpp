#include "Global.h"
#include "MatrixMultiplicationUnitTests.h"
#include "FullyConnectedLayerKernelGPU.h"
#include "CostFunctions.h"
#include "CostFunctionsGPU.h"
#include "ActivationFunctions.h"
#include <QDebug>
#include <cuda.h>
#include <cuda_runtime.h>

static void matrixMultiplyTest(const int batchSize, const int inputSize, const int outputSize)
{
    float *A = new float[inputSize * outputSize];
    float *B = new float[batchSize * inputSize];
    float *C = new float[batchSize * outputSize];

    for(int i = 0; i < inputSize * outputSize;++i)
    {
        A[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

        //A[i] = i;
    }

    for (int i =0;i<batchSize * inputSize; ++i)
    {
        B[i] = static_cast <double> (rand()) /static_cast <double> (RAND_MAX);
        //B[i] = i;
    }

    for(int y = 0; y < batchSize; ++y)
    {
        for(int x = 0; x< outputSize; ++x)
        {
            C[y*outputSize + x] = 0.0;
            for(int i = 0; i< inputSize;i++)
            {
                C[y*outputSize+x] += B[y*inputSize + i] * A[i*outputSize+x];
            }
        }
    }

    float *A_gpu = nullptr;
    cudaMalloc(&A_gpu, inputSize * outputSize*sizeof(float));

    float *B_gpu = nullptr;
    cudaMalloc(&B_gpu, batchSize * inputSize*sizeof(float));

    float *C_gpu = nullptr;
    cudaMalloc(&C_gpu, batchSize * outputSize*sizeof(float));

    cudaMemcpy(A_gpu, A, inputSize * outputSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, batchSize * inputSize*sizeof(float), cudaMemcpyHostToDevice);

    float (*activation) (const float) = nullptr;

    activation = (float(*)(float))getActivationFuncFloat(None);

    FullyConnectedLayerKernelGPU(A_gpu, batchSize, B_gpu, inputSize, C_gpu, outputSize, activation);

    float *C_readback = new float[batchSize * outputSize];

    cudaMemcpy(C_readback, C_gpu, batchSize*outputSize*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0;i< batchSize*outputSize;++i)
    {
        float diff = C_readback[i] - C[i];
       QCOMPARE(diff*diff < 0.001, true);
      // qDebug() << C[i] << C_readback[i];
    }

    delete [] A;
    delete [] B;
    delete [] C;
    delete [] C_readback;

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);


}

static void testCostFunction(const int batchSize, const int outputSize, CostFunction costFunction)
{
    float *lastActivations = new float[batchSize * outputSize];
    float *labels = new float[batchSize * outputSize];

    for(int e = 0; e < batchSize; ++e)
    {
        for(int i = 0;i<outputSize;++i)
        {
            lastActivations[e*outputSize + i] = static_cast <double> (rand()) /static_cast <double> (RAND_MAX);
            labels[e*outputSize + i] = static_cast <double> (rand()) /static_cast <double> (RAND_MAX);
        }
    }

    float *lastActivations_gpu = nullptr;
    float *labels_gpu = nullptr;

    cudaMalloc(&lastActivations_gpu, sizeof(float) * outputSize * batchSize);
    cudaMalloc(&labels_gpu, sizeof(float) * outputSize * batchSize);

    cudaMemcpy(lastActivations_gpu, lastActivations, sizeof(float) * outputSize * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(labels_gpu,labels, sizeof(float)*outputSize*batchSize, cudaMemcpyHostToDevice);

    float *cost = new float[batchSize];
    float *derivatives = new float[batchSize * outputSize];

    float *cost_gpu = nullptr;
    float *derivatives_gpu = nullptr;

    cudaMalloc(&cost_gpu, sizeof(float) * batchSize);
    cudaMalloc(&derivatives_gpu, sizeof(float) * batchSize * outputSize);

    if (costFunction == CrossEntropy)
    {
        crossEntropySigmoidCPU<float>(lastActivations, outputSize, labels, cost, batchSize, derivatives);
        crossEntropySigmoidGPUKernel(lastActivations_gpu,outputSize,labels_gpu,cost_gpu, batchSize, derivatives_gpu);
    }
    else if (costFunction == MeanSquared)
    {
        meanSquaredRectifierCPU<float>(lastActivations, outputSize, labels, cost, batchSize, derivatives);
        meanSquaredRectifierGPUKernel(lastActivations_gpu,outputSize,labels_gpu,cost_gpu, batchSize, derivatives_gpu);
    }

    float *cost_gpu_readback = new float[batchSize];
    float *derivatives_gpu_readback= new float[batchSize * outputSize];

    cudaMemcpy(cost_gpu_readback, cost_gpu, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(derivatives_gpu_readback, derivatives_gpu, sizeof(float) * outputSize * batchSize, cudaMemcpyDeviceToHost);

    for(int i = 0;i< batchSize; ++i)
    {
        float diff = cost_gpu_readback[i] - cost[i];
        QCOMPARE(diff*diff < 0.001, true);
    }

    for(int i = 0;i< batchSize*outputSize;++i)
    {
        float diff = derivatives_gpu_readback[i] - derivatives[i];
        QCOMPARE(diff*diff < 0.001, true);
    }

    cudaFree(cost_gpu);
    cudaFree(derivatives_gpu);
    cudaFree(lastActivations_gpu);
    cudaFree(labels_gpu);

    delete [] lastActivations;
    delete [] labels;
    delete [] cost;
    delete [] derivatives;
    delete [] cost_gpu_readback;
    delete [] derivatives_gpu_readback;

}

static void gradientCheckOnCostFunction(const int outputSize, CostFunction costFunction)
{
    const int batchSize = 1;
    const float epsilon = 0.004;

    float *last = new float[batchSize * outputSize];
    float *lastActivations = new float[batchSize * outputSize];
    float *labels = new float[batchSize * outputSize];

    for(int i = 0;i<outputSize;++i)
    {
        last[i] = static_cast <double> (rand()) /static_cast <double> (RAND_MAX);
        labels[i] = static_cast <double> (rand()) /static_cast <double> (RAND_MAX);
    }

    if (costFunction == CrossEntropy)
    {
        for(int i = 0;i<outputSize;++i)
        {
            lastActivations[i] = sigmoid<float>(last[i]);
        }
    }
    else if (costFunction == MeanSquared)
    {
        for(int i = 0;i<outputSize;++i)
        {
            lastActivations[i] = rectifier<float> (last[i]);
        }
    }

    float cost = 0;
    float *derivatives = new float[outputSize];
    float *fakeDerivatives = new float[outputSize];

    if (costFunction == CrossEntropy)
    {
        crossEntropySigmoidCPU<float>(lastActivations, outputSize, labels, &cost, batchSize, derivatives);
    }
    else if (costFunction == MeanSquared)
    {
        meanSquaredRectifierCPU<float>(lastActivations, outputSize, labels, &cost, batchSize, derivatives);
    }


    for(int i = 0; i<outputSize;++i)
    {
        float x = last[i];
        float xPlusE = x + epsilon;
        float xMinusE = x - epsilon;

        float costPlus = 0.0;
        float costMinus = 0.0;

        float oldActivation = lastActivations[i];

        if (costFunction == CrossEntropy)
        {


            lastActivations[i] =  sigmoid<float>(xPlusE);
            crossEntropySigmoidCPU<float>(lastActivations, outputSize, labels, &costPlus, batchSize, fakeDerivatives);

            lastActivations[i] = sigmoid<float>(xMinusE);
            crossEntropySigmoidCPU<float>(lastActivations, outputSize, labels, &costMinus, batchSize, fakeDerivatives);

        }
        else if (costFunction == MeanSquared)
        {
            lastActivations[i] = rectifier<float>(xPlusE);

            meanSquaredRectifierCPU<float>(lastActivations, outputSize, labels, &costPlus, batchSize, derivatives);

            lastActivations[i] = rectifier<float>(xMinusE);

            meanSquaredRectifierCPU<float>(lastActivations, outputSize, labels, &costMinus, batchSize, derivatives);
        }

        lastActivations[i] = oldActivation;

        float gradient = (costPlus - costMinus) / (2.0 * epsilon);

        float diff = derivatives[i] - gradient;
        QCOMPARE(diff*diff < 0.001, true);

    }




    delete [] lastActivations;
    delete [] labels;
    delete [] derivatives;
    delete [] fakeDerivatives;
    delete [] last;

}



void MatrixMultiplicationUnitTests::initTestCase()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
        printf("Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Device texture alignment: %lu\n", deviceProp.textureAlignment);
        printf("Device texture dimension: %d X %d\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
    }

    size_t freeMem = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("available video memory: %ld, %ld (bytes)\n", freeMem, totalMem);

    srand(0);
}

void MatrixMultiplicationUnitTests::testSmallMatrix()
{
    matrixMultiplyTest(3, 3, 32);
}

void MatrixMultiplicationUnitTests::testLargeMatrix()
{
    const int inputSize = 1032;
    const int batchSize = 450;
    const int outputSize = 678;

    matrixMultiplyTest(batchSize, inputSize, outputSize);
}

void MatrixMultiplicationUnitTests::testSpecialMatrix_x_32()
{
    const int inputSize = 1032;
    const int batchSize = 450;
    const int outputSize = 64;
    matrixMultiplyTest(batchSize, inputSize, outputSize);

}

void MatrixMultiplicationUnitTests::testSpecialMatrix_32_x()
{
    const int inputSize = 64;
    const int batchSize = 450;
    const int outputSize = 678;
    matrixMultiplyTest(batchSize, inputSize, outputSize);

}

void MatrixMultiplicationUnitTests::testSpecialMatrix_32_32()
{
    const int inputSize = 64;
    const int batchSize = 450;
    const int outputSize = 320;
    matrixMultiplyTest(batchSize, inputSize, outputSize);

}

void MatrixMultiplicationUnitTests::testCostFunctionCPUvsGPUCrossEntropySigmoid()
{
    const int outputSize = 64;
    const int batchSize = 450;

    testCostFunction(batchSize, outputSize, CrossEntropy);
}

void MatrixMultiplicationUnitTests::testCostFunctionCPUvsGPUMeanSquaredRectifier()
{
    const int outputSize = 64;
    const int batchSize = 450;

    testCostFunction(batchSize, outputSize, MeanSquared);
}

void MatrixMultiplicationUnitTests::gradientCheckOnCrossEntropySigmoidCost()
{
    const int outputSize = 64;
    gradientCheckOnCostFunction(outputSize, CrossEntropy);
}

void MatrixMultiplicationUnitTests::gradientCheckOnMeanSquaredRectifierCost()
{
    const int outputSize = 64;
    gradientCheckOnCostFunction(outputSize, MeanSquared);
}

QTEST_MAIN(MatrixMultiplicationUnitTests)

