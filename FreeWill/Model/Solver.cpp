#include "Solver.h"
#include "Model.h"
#include <sstream>

bool FreeWill::Solver::init(FreeWill::Model *model)
{
    clearUpdateOperators();
    for(auto iter = model->m_updatePairs.begin(); iter != model->m_updatePairs.end(); ++iter)
    {
        FreeWill::TensorDescriptorHandle operandA = iter->first;
        FreeWill::TensorDescriptorHandle operandB = iter->second;

        std::stringstream operatorNameStream;

        DataType dataType = model->m_tensors[operandA.first]->m_dataType;

        operatorNameStream<<"Update_"<<operandA.first<<"_with_"<<operandB.first;

        addUpdateOperator(operatorNameStream.str(), FreeWill::ELEMENTWISE_ADD,
                            {{"OperandA", operandA}, {"OperandB", operandB}},
                            {{"Result", operandA}},{}, dataType);
    }

    for(auto iter = m_updateOperators.begin(); iter != m_updateOperators.end(); ++iter)
    {
        switch(m_deviceUsed)
        {
        case FreeWill::DeviceType::CPU_NAIVE:
        (*iter)->init<FreeWill::DeviceType::CPU_NAIVE>(model->m_tensors);
            break;
        case FreeWill::DeviceType::GPU_CUDA:
        (*iter)->init<FreeWill::DeviceType::GPU_CUDA>(model->m_tensors);
            break;
        }
    }

    return true;
}

FreeWill::OperatorDescriptorHandle FreeWill::Solver::addUpdateOperator(const std::string &name,
                                 FreeWill::OperatorName operatorName,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType)
{
    FreeWill::OperatorDescriptor *opDescriptor = new FreeWill::OperatorDescriptor(name, operatorName, inputs, outputs, properties, dataType);

    m_updateOperators.push_back(opDescriptor);

    return name;
}

void FreeWill::Solver::clearUpdateOperators()
{
    for (unsigned int i = 0; i<m_updateOperators.size();++i)
    {
        delete m_updateOperators[i];
    }
    m_updateOperators.clear();
}


void FreeWill::Solver::forward(FreeWill::Model *model)
{
    auto iter = model->m_forwardPath.begin();

    switch(m_deviceUsed)
    {
    case FreeWill::DeviceType::CPU_NAIVE:
        for(; iter != model->m_forwardPath.end();++iter)
        {
            model->m_operators[(*iter)]->evaluate<FreeWill::DeviceType::CPU_NAIVE>();
        }
        break;
    case FreeWill::DeviceType::GPU_CUDA:
        for(; iter != model->m_forwardPath.end();++iter)
        {
            model->m_operators[(*iter)]->evaluate<FreeWill::DeviceType::GPU_CUDA>();
        }
        break;
    }
}

void FreeWill::Solver::backward(FreeWill::Model *model)
{
    auto iter = model->m_backwardPath.begin();

    switch(m_deviceUsed)
    {
    case FreeWill::DeviceType::CPU_NAIVE:
        for(; iter != model->m_backwardPath.end();++iter)
        {
            model->m_operators[(*iter)]->evaluate<FreeWill::DeviceType::CPU_NAIVE>();
        }
        break;
    case FreeWill::DeviceType::GPU_CUDA:
        for(; iter != model->m_backwardPath.end();++iter)
        {
            model->m_operators[(*iter)]->evaluate<FreeWill::DeviceType::GPU_CUDA>();
        }
        break;
    }
}

void FreeWill::Solver::update(double learningRate)
{
    if (m_previousLearningRate != learningRate)
    {
        for(auto iter = m_updateOperators.begin(); iter != m_updateOperators.end(); ++iter)
        {
            switch(m_deviceUsed)
            {
            case FreeWill::DeviceType::CPU_NAIVE:
                (*iter)->evaluateWithParameterUpdate<FreeWill::DeviceType::CPU_NAIVE>({{"Rate", learningRate}});
                break;
            case FreeWill::DeviceType::GPU_CUDA:
                (*iter)->evaluateWithParameterUpdate<FreeWill::DeviceType::GPU_CUDA>({{"Rate", learningRate}});
                break;
            }
        }

        m_previousLearningRate = learningRate;
    }
    else
    {
        for(auto iter = m_updateOperators.begin(); iter != m_updateOperators.end(); ++iter)
        {
            switch(m_deviceUsed)
            {
            case FreeWill::DeviceType::CPU_NAIVE:
                (*iter)->evaluate<FreeWill::DeviceType::CPU_NAIVE>();
                break;
            case FreeWill::DeviceType::GPU_CUDA:
                (*iter)->evaluate<FreeWill::DeviceType::GPU_CUDA>();
                break;
            }
        }
    }
}

FreeWill::Solver::Solver()
    :m_previousLearningRate(0.0)
{}

FreeWill::Solver::~Solver()
{
    clearUpdateOperators();
}
