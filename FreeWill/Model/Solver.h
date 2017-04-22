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
        std::vector<OperatorDescriptor*> m_updateOperators;
        double m_previousLearningRate;
    public:
        DeviceType m_deviceUsed;
        unsigned int m_batchSize;

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
