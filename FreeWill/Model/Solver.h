#ifndef SOLVER_H
#define SOLVER_H

#include "../DeviceSelection.h"
#include "../Operator/Operator.h"
#include <vector>
#include "OperatorDescriptor.h"

namespace FreeWill
{
    class Model;
    class Solver
    {
        //std::vector<OperatorDescriptor*> m_updateOperators;
        std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> m_mergeGradientOperators;
        std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> m_updateFirstDeviceTensorOperators;
        std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> m_broadcastTensorToSiblingOperators;

        double m_previousLearningRate;
    public:
        DeviceType m_deviceUsed;
        unsigned int m_batchSize;
        DataType m_dataType;

        bool init(Model *model);

        void forward(Model *model);
        void backward(Model *model);

        void update(double learningRate = -0.01);

        Solver();

        ~Solver();

    private:
        OperatorDescriptorHandle addUpdateOperator(const std::string &name,
                                 FreeWill::OperatorName operatorName,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType);

        void clearUpdateOperators();
    };
}

#endif
