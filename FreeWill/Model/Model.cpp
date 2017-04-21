#include "Model.h"
#include <cmath>
#include "../Operator/Operator.h"
#include <fstream>
#include <sstream>

FreeWill::Model* FreeWill::Model::create()
{
    FreeWill::Model *model = new FreeWill::Model();
    return model;
}

FreeWill::TensorDescriptorHandle FreeWill::Model::addTensor(const std::string &name, const Shape &shape, bool isBatchTensor, DataType dataType)
{
    if (m_tensors.find(name) != m_tensors.end())
    {
        return {std::string(), Shape()};
    }
   
    m_tensors[name] = new FreeWill::TensorDescriptor(name, shape, isBatchTensor, dataType);
    
   return {name, Shape()};
}



FreeWill::Model::Model()
    :m_tensors(),
      m_operators()
{
}

int FreeWill::Model::addOperator(const std::string &name,
                                 const std::string &operatorNameString,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType)
{
    if (FreeWill::operatorNameTable.find(operatorNameString) != FreeWill::operatorNameTable.end())
    {
        FreeWill::OperatorName operatorName = operatorNameTable[operatorNameString];

        addOperator(name, operatorName, inputs, outputs, properties, dataType);
    }

    return -1;
}

int FreeWill::Model::addOperator(const std::string &name,
                                 FreeWill::OperatorName operatorName,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType)
{
    if (m_operators.find(name) != m_operators.end())
    {
        return -1;
    }

    OperatorDescriptor *opDescriptor = new OperatorDescriptor(name, operatorName, inputs, outputs, properties, dataType);

    m_operators[name] = opDescriptor;

    return m_operators.size() - 1;
}



bool FreeWill::Model::init(const Solver &solver)
{
    //allocating tensors
    std::map<std::string, TensorDescriptor*>::iterator iterTensor = m_tensors.begin();

    //creating operators

    std::map<std::string, OperatorDescriptor*>::iterator iterOperator = m_operators.begin();

    switch (solver.m_deviceUsed)
    {
    case CPU_NAIVE:
        for(;iterTensor != m_tensors.end(); ++iterTensor)
        {
            std::cout << iterTensor->first << std::endl;
            TensorDescriptor *descriptor = iterTensor->second;
            descriptor->allocateTensor<FreeWill::CPU_NAIVE>(solver);
        }

        for(;iterOperator != m_operators.end(); ++iterOperator)
        {
            std::cout << iterOperator->first << std::endl;
            OperatorDescriptor *descriptor = iterOperator->second;

            if (!descriptor->init<FreeWill::CPU_NAIVE>(m_tensors))
            {
                return false;
            }
        }

        break;
    case GPU_CUDA:

        for(;iterTensor != m_tensors.end(); ++iterTensor)
        {
            std::cout << iterTensor->first << std::endl;
            TensorDescriptor *descriptor = iterTensor->second;
            descriptor->allocateTensor<FreeWill::GPU_CUDA>(solver);
        }

        for(;iterOperator != m_operators.end(); ++iterOperator)
        {
            std::cout << iterOperator->first << std::endl;
            OperatorDescriptor *descriptor = iterOperator->second;

            if (!descriptor->init<FreeWill::GPU_CUDA>(m_tensors))
            {
                std::cout << iterOperator->first << "sanity check failed!" << std::endl;
                return false;
            }
        }

        break;
    default:
        return false;
    }

    return true;
}

bool FreeWill::Model::defineForwardPath(const std::vector<std::string> &forwardOperators)
{
    m_forwardPath.clear();

    for(unsigned int i = 0; i<forwardOperators.size();++i)
    {
        if (m_operators.find(forwardOperators[i]) != m_operators.end())
        {
            m_forwardPath.push_back(forwardOperators[i]);
        }
        else
        {
            m_forwardPath.clear();
            return false;
        }
    }

    return true;
}

void FreeWill::Model::generateSVGDiagram(const std::string &filename)
{
    std::stringstream tempStream;

    unsigned int width = 0;
    unsigned int height = 0;

    for(unsigned int i = 0;i<m_forwardPath.size();++i)
    {
        unsigned int operatorWidth = 0;
        unsigned int operatorHeight = 0;

        m_operators[m_forwardPath[i]]->evaluateSVGDiagramSize(operatorWidth, operatorHeight);

        width += operatorWidth;

        if (height < operatorHeight)
        {
            height = operatorHeight;
        }

    }

    unsigned int offset = 0;

    for(unsigned int i = 0;i<m_forwardPath.size();++i)
    {
        unsigned int operatorWidth = 0;
        unsigned int operatorHeight = 0;

        std::stringstream ss;
        m_operators[m_forwardPath[i]]->generateSVGDiagram(ss, operatorWidth, operatorHeight);

        tempStream << "<g transform=\"translate(" << offset << "," << 0.5f * (height - operatorHeight) << ")\">";
        tempStream << ss.str();
        tempStream << "</g>";

        offset += operatorWidth;

        if (height < operatorHeight)
        {
            height = operatorHeight;
        }
    }



    std::ofstream outputStream;

    outputStream.open(filename);

    outputStream << "<svg width=\"" << width << "\" height=\"" << height << "\" viewBox=\"0 0 "<< width <<" "<< height << "\" xmlns=\"http://www.w3.org/2000/svg\">";

    outputStream << tempStream.str();

    outputStream << "</svg>";

    outputStream.close();
}
