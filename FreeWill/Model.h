#ifndef MODEL_H
#define MODEL_H

#include "DeviceSelection.h"
#include <vector>
#include "Tensor/Tensor.h"
#include "Operator/Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed, typename DataType>
    class Model
    {
    private:
       // std::vector<Tensor<DeviceUsed, DataType>> m_tensors;
        //std::vector<Operator *> m_operators;

    public:
        Model(){}
        bool build(){}



    };
}


#endif
