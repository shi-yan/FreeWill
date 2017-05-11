#include "Solver.h"
#include "Model.h"
#include <sstream>

bool FreeWill::Solver::init(FreeWill::Model *model)
{
    if (!model->init(*this))
    {
        return false;
    }

    clearUpdateOperators();

    m_dataType = model->m_tensors[model->m_updatePairs.begin()->second.first]->m_dataType;

    for(auto iter = model->m_updatePairs.begin(); iter != model->m_updatePairs.end(); ++iter)
    {
        FreeWill::TensorDescriptorHandle operandB = iter->second;

        model->generateGradientMergeOperators(m_mergeGradientOperators, operandB);

    }

    for(auto iter = model->m_updatePairs.begin(); iter != model->m_updatePairs.end(); ++iter)
    {
        FreeWill::TensorDescriptorHandle operandA = iter->first;
        FreeWill::TensorDescriptorHandle operandB = iter->second;

        //std::stringstream operatorNameStream;

        //DataType dataType = model->m_tensors[operandA.first]->m_dataType;

        //operatorNameStream<<"Update_"<<operandA.first<<"_with_"<<operandB.first;

        /*addUpdateOperator(operatorNameStream.str(), FreeWill::OperatorName::ELEMENTWISE_ADD,
                            {{"OperandA", operandA}, {"OperandB", operandB}},
                            {{"Result", operandA}},{}, dataType);*/

        model->generateUpdateFirstDeviceTensorOperators(m_updateFirstDeviceTensorOperators, operandA, operandB);


    }

    for(auto iter = model->m_updatePairs.begin(); iter != model->m_updatePairs.end(); ++iter)
    {
        FreeWill::TensorDescriptorHandle operandA = iter->first;

        model->generateBroadcastFirstDeviceTensorOperators(m_broadcastTensorToSiblingOperators, operandA);
    }

    /*for(auto iter = m_updateOperators.begin(); iter != m_updateOperators.end(); ++iter)
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
    }*/

    return true;
}
/*
FreeWill::OperatorDescriptorHandle FreeWill::Solver::addUpdateOperator(const std::string &name,
                                 FreeWill::OperatorName operatorName,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType)
{
    FreeWill::OperatorDescriptor *opDescriptor = new FreeWill::OperatorDescriptor(name, operatorName, inputs, outputs, properties, dataType);

    m_updateOperators.push_back(opDescriptor);

    return name;
}*/

void FreeWill::Solver::clearUpdateOperators()
{
    /*for (unsigned int i = 0; i<m_updateOperators.size();++i)
    {
        delete m_updateOperators[i];
    }
    m_updateOperators.clear();*/

    /*std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> m_mergeGradientOperators;
    std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> m_updateFirstDeviceTensorOperators;
    std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> m_broadcastTensorToSiblingOperators;
    */

    for (unsigned int i = 0; i<m_mergeGradientOperators.size(); ++i)
    {
        switch(m_deviceUsed)
        {
        case FreeWill::DeviceType::CPU_NAIVE:
            delete std::get<Operator<DeviceType::CPU_NAIVE>*>(m_mergeGradientOperators[i]);
            break;
        case FreeWill::DeviceType::GPU_CUDA:
            delete std::get<Operator<DeviceType::GPU_CUDA>*>(m_mergeGradientOperators[i]);
            break;
        }
    }

    for (unsigned int i =0;i<m_updateFirstDeviceTensorOperators.size();++i)
    {
        switch(m_deviceUsed)
        {
        case FreeWill::DeviceType::CPU_NAIVE:
            delete std::get<Operator<DeviceType::CPU_NAIVE>*>(m_updateFirstDeviceTensorOperators[i]);
            break;
        case FreeWill::DeviceType::GPU_CUDA:
            delete std::get<Operator<DeviceType::GPU_CUDA>*>(m_updateFirstDeviceTensorOperators[i]);
            break;
        }
    }

    for (unsigned int i =0;i<m_broadcastTensorToSiblingOperators.size();++i)
    {
        switch (m_deviceUsed)
        {
        case FreeWill::DeviceType::CPU_NAIVE:
            delete std::get<Operator<DeviceType::CPU_NAIVE>*>(m_broadcastTensorToSiblingOperators[i]);
            break;
        case FreeWill::DeviceType::GPU_CUDA:
            delete std::get<Operator<DeviceType::GPU_CUDA>*>(m_broadcastTensorToSiblingOperators[i]);
            break;
        default:
            break;
        }

    }
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
            //break;
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
    for(unsigned int i = 0;i<m_mergeGradientOperators.size();++i)
    {
        switch(m_deviceUsed)
        {
        case FreeWill::DeviceType::CPU_NAIVE:
            std::get<Operator<FreeWill::DeviceType::CPU_NAIVE>*>(m_mergeGradientOperators[i])->evaluate();
            break;
        case FreeWill::DeviceType::GPU_CUDA:
            std::get<Operator<FreeWill::DeviceType::GPU_CUDA>*>(m_mergeGradientOperators[i])->evaluate();
            break;
        }
    }


    /*if (m_previousLearningRate != learningRate)
    {
        for(auto iter = m_updateOperators.begin(); iter != m_updateOperators.end(); ++iter)
        {
            switch(m_deviceUsed)
            {
            case FreeWill::DeviceType::CPU_NAIVE:
                (*iter)->evaluateWithParameterUpdate<FreeWill::DeviceType::CPU_NAIVE>({{"Rate", (float)learningRate}});
                break;
            case FreeWill::DeviceType::GPU_CUDA:
                (*iter)->evaluateWithParameterUpdate<FreeWill::DeviceType::GPU_CUDA>({{"Rate", (float)learningRate}});
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
    }*/



    for(auto iter = m_updateFirstDeviceTensorOperators.begin(); iter != m_updateFirstDeviceTensorOperators.end(); ++iter)
    {
        switch(m_deviceUsed)
        {
        case DeviceType::CPU_NAIVE:
        {
            Operator<DeviceType::CPU_NAIVE> *operatorBase = std::get<Operator<DeviceType::CPU_NAIVE>*>(*iter);
            if (m_dataType == DataType::FLOAT)
            {
                ElementwiseAdd<DeviceType::CPU_NAIVE, float> *elementwiseAdd = dynamic_cast<ElementwiseAdd<DeviceType::CPU_NAIVE, float>*>(operatorBase);
                elementwiseAdd->setRate(learningRate);
            }
            else if(m_dataType == DataType::DOUBLE)
            {
                ElementwiseAdd<DeviceType::CPU_NAIVE, double> *elementwiseAdd = dynamic_cast<ElementwiseAdd<DeviceType::CPU_NAIVE, double>*>(operatorBase);
                elementwiseAdd->setRate(learningRate);
            }
            operatorBase->evaluate();
        }
            break;
        case DeviceType::GPU_CUDA:
        {
            Operator<DeviceType::GPU_CUDA> *operatorBase = std::get<Operator<DeviceType::GPU_CUDA>*>(*iter);
            if (m_dataType == DataType::FLOAT)
            {
                ElementwiseAdd<DeviceType::GPU_CUDA, float> *elementwiseAdd = dynamic_cast<ElementwiseAdd<DeviceType::GPU_CUDA, float>*>(operatorBase);
                elementwiseAdd->setRate(learningRate);
            }
            else if(m_dataType == DataType::DOUBLE)
            {
                ElementwiseAdd<DeviceType::GPU_CUDA, double> *elementwiseAdd = dynamic_cast<ElementwiseAdd<DeviceType::GPU_CUDA, double> *>(operatorBase);
                elementwiseAdd->setRate(learningRate);
            }
            operatorBase->evaluate();
        }
            break;
        }
    }

    for(auto iter = m_broadcastTensorToSiblingOperators.begin(); iter != m_broadcastTensorToSiblingOperators.end(); ++iter)
    {
        switch (m_deviceUsed)
        {
        case DeviceType::CPU_NAIVE:
        {
            Operator<DeviceType::CPU_NAIVE> *operatorBase = std::get<Operator<DeviceType::CPU_NAIVE>*>(*iter);
            operatorBase->evaluate();

        }


            break;
        case DeviceType::GPU_CUDA:
        {
            Operator<DeviceType::GPU_CUDA> *operatorBase = std::get<Operator<DeviceType::GPU_CUDA>*>(*iter);
            operatorBase->evaluate();
        }
            break;
        default:
            break;
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
