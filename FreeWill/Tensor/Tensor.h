#ifndef TENSOR_H
#define TENSOR_H

#include <type_traits>
#include <string>
#include <memory>
#include "DeviceSelection.h"
#include "Shape.h"
#include "ReferenceCountedBlob.h"
#include <ctime>
#include <cuda.h>
#include <cudnn.h>
#include "../Context/Context.h"

namespace FreeWill 
{

    template<DeviceType DeviceUsed, typename DataType>
    class Tensor;

    template<DeviceType DeviceUsed>
    class TensorBase
    {
    protected:

       Shape m_shape;
       cudnnTensorDescriptor_t m_gpuTensorDescriptor;
       ReferenceCountedBlob<DeviceUsed> m_data;

       TensorBase(const Shape &shape = Shape()) 
           :m_shape(shape),
            m_gpuTensorDescriptor(0),
            m_data()
       {
           RUN_CUDNN(cudnnCreateTensorDescriptor(&m_gpuTensorDescriptor));
       }

       TensorBase(const ReferenceCountedBlob<DeviceUsed> &data, const Shape &shape = Shape())
           :m_shape(shape),
               m_data(data),
               m_gpuTensorDescriptor(0)
       {
           RUN_CUDNN(cudnnCreateTensorDescriptor(&m_gpuTensorDescriptor));
       }

       void *gpuDataHandle()
       {
            return m_data.m_gpuDataHandle;
       }

       void *cpuDataHandle()
       {
            return m_data.m_dataHandle;
       }

    public:
       const cudnnTensorDescriptor_t &gpuTensorDescriptor() const
       {
           return m_gpuTensorDescriptor;
       }

       virtual const Shape &shape() const
       {
            return m_shape;
       }

       void clear()
       {
            m_data.clear();
       }

       virtual ~TensorBase() 
       {
           RUN_CUDNN(cudnnDestroyTensorDescriptor(m_gpuTensorDescriptor));
       }

       template<typename DataType = float>
       Tensor<DeviceUsed, DataType> *toType()
       {
            return dynamic_cast< Tensor<DeviceUsed, DataType> *>(this);
       }
    };
    
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Tensor : public TensorBase<DeviceUsed>
    {
    private:
        using TensorBase<DeviceUsed>::m_shape;
        std::string m_name;
        using TensorBase<DeviceUsed>::m_data;
        
    public:
        using TensorBase<DeviceUsed>::shape;

        explicit Tensor(const std::initializer_list<unsigned int> &shape, 
                const std::string &name = "no_name")
            :TensorBase<DeviceUsed>(shape),
            m_name(name)
        {

        }
        
        explicit Tensor(const Shape &shape = Shape(),
	       const std::string &name = "no_name")
            :TensorBase<DeviceUsed>(shape),
            m_name(name)
	    {
	    }

        explicit Tensor(const Tensor &in)
            :TensorBase<DeviceUsed>(in.m_data, in.shape()),
            m_name(in.m_name)
        {
        }

        bool init()
	    {
            unsigned int size = m_shape.size();
            bool result = false;
            if (size) 
            {
                result = m_data.alloc(size * sizeof(DataType));
            }

            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                updateGPUTensorDescriptor();
            }

            return result;
	    }

        bool init(const std::initializer_list<DataType> &initList)
        {
            if (!init())
            {
                return false;
            }

            unsigned int size = m_shape.size();
            std::copy(initList.begin(), initList.begin() + (initList.size()>size?size:initList.size()), (DataType*) m_data.dataHandle());

            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                m_data.copyFromHostToDevice();
                updateGPUTensorDescriptor();
            }

            return true;
        }

        void randomize()
        {
           //std::random_device rd;
           static std::mt19937 gen(/*rd()*/ std::time(NULL));
           //std::uniform_real_distribution<DataType> dis(0, 1);
           std::normal_distribution<DataType> normDis(0, 1);
           DataType *bits = (DataType *) m_data.dataHandle();
           unsigned int size = m_shape.size();
                 
           for (unsigned int n = 0; n < size; ++n) 
           {
                bits[n] = normDis(gen);
                //bits[n] = ((double) rand() / (double) RAND_MAX - 0.5) * 0.1;
            } 
 
            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                m_data.copyFromHostToDevice();
            }
            
        }

        void operator=(const Tensor<DeviceUsed, DataType> &in)
        {
            m_shape = in.m_shape;
            m_name = in.m_name;
            m_data = in.m_data;
            updateGPUTensorDescriptor();
        }

        DataType &operator[](unsigned int i)
        {
            DataType *bits = (DataType *) m_data.dataHandle();
            return *(bits + i);
        }

        void reshape(const Shape &newShape)
        {
            if (newShape.size() == m_shape.size())
            {
                m_shape = newShape;
                updateGPUTensorDescriptor();
            }
        }

        void copyFromDeviceToHost()
        {
            m_data.copyFromDeviceToHost();
        }

        void copyFromHostToDevice()
        {
            m_data.copyFromHostToDevice();
        }
        
        DataType *gpuDataHandle()
        {
            return (DataType*) TensorBase<DeviceUsed>::gpuDataHandle();
        }

        DataType *cpuDataHandle()
        {
            return (DataType*) TensorBase<DeviceUsed>::cpuDataHandle();
        }

    private:
        void updateGPUTensorDescriptor()
        {
            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
                if constexpr (std::is_same<DataType,float>::value)
                {
                    dataType = CUDNN_DATA_FLOAT;
                }
                else if constexpr (std::is_same<DataType,double>::value)
                {
                    dataType = CUDNN_DATA_DOUBLE;
                }

                int nbDims = m_shape.dimension();
                int atLeastDims = nbDims < 4 ? 4 : nbDims;
                int *dimA = new int[atLeastDims];
                int *strideA = new int[atLeastDims];
                
                for(int i = 0;i<atLeastDims;++i)
                {
                    if (i < nbDims)
                    {
                        dimA[atLeastDims - i - 1] = m_shape[i];
                        if (i == 0)
                        {
                            strideA[atLeastDims-1] = 1;
                        }
                        else
                        {
                            strideA[atLeastDims - 1 - i] = m_shape[i-1] * strideA[atLeastDims - i];
                        }
                    }
                    else
                    {
                        dimA[atLeastDims - i - 1] = 1;
                        strideA[atLeastDims - 1 - i] = strideA[atLeastDims - i];
                    }
                }
               
                //printf("create tensor descriptor: %d %d %d\n", nbDims,dimA[0], strideA[0]);
                // cudnn only supports dim >= 4, it seems that to use cudnn activation, dim has to be >= 4
                RUN_CUDNN(cudnnSetTensorNdDescriptor(TensorBase<DeviceUsed>::m_gpuTensorDescriptor,
                                           dataType,
                                           atLeastDims,
                                           dimA,
                                           strideA));
                delete [] dimA;
                delete [] strideA;
            }
        }
    };

}

#endif
