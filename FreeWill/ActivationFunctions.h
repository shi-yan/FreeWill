
#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <cuda.h>
#include <cuda_runtime.h>

template<class ScalarType>
__device__ ScalarType sigmoid(const ScalarType in)
{
    return 1.0 / (1.0 + exp(-in));
}

static __device__ float (*fptr_sigmoid) (const float)  = sigmoid<float>;
static __device__ double (*fptr_sigmoid_d) (const double)  = sigmoid<double>;

template<class ScalarType>
__device__ ScalarType sigmoidDerivative(const ScalarType in)
{
    return in * (1.0 - in);
}

static __device__ float (*fptr_sigmoidDerivative) (const float)  = sigmoidDerivative<float>;
static __device__ double (*fptr_sigmoidDerivative_d) (const double)  = sigmoidDerivative<double>;

template<class ScalarType>
__device__ ScalarType rectifier(const ScalarType in)
{
     return in > 0 ? in : 0.0;
}

static __device__ float (*fptr_rectifier) (const float)  = rectifier<float>;
static __device__ double (*fptr_rectifier_d) (const double)  = rectifier<double>;

template<class ScalarType>
__device__ ScalarType rectifierDerivative(const ScalarType in)
{
    return in > 0.0 ? 1.0 : 0.0;
}


static __device__ float (*fptr_rectifierDerivative) (const float)  = rectifierDerivative<float>;
static __device__ double (*fptr_rectifierDerivative_d) (const double)  = rectifierDerivative<double>;

template<class ScalarType>
__device__ ScalarType tanh(const ScalarType in)
{
    ScalarType y = exp(2.0 * in);
    return (y - 1) / (y + 1);
}


static __device__ float (*fptr_tanh) (const float)  = tanh<float>;
static __device__ double (*fptr_tanh_d) (const double)  = tanh<double>;

template<class ScalarType>
__device__ ScalarType tanhDerivative(const ScalarType in)
{
    return (1.0 - in * in);
}


static __device__ float (*fptr_tanhDerivative) (const float)  = tanhDerivative<float>;
static __device__ double (*fptr_tanhDerivative_d) (const double)  = tanhDerivative<double>;

template<class ScalarType>
__device__ ScalarType noActivation(const ScalarType in)
{
    return in;
}

static __device__ float (*fptr_noActivation) (const float)  = noActivation<float>;
static __device__ double (*fptr_noActivation_d) (const double)  = noActivation<double>;

template<class ScalarType>
__device__ ScalarType noActivationDerivative(const ScalarType in)
{
    return 1.0;
}


static __device__ float (*fptr_noActivationDerivative) (const float)  = noActivationDerivative<float>;
static __device__ double (*fptr_noActivationDerivative_d) (const double)  = noActivationDerivative<double>;

#endif
