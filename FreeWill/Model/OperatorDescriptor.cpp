#include "OperatorDescriptor.h"


FreeWill::OperatorDescriptor::OperatorDescriptor(const std::string &name,
        FreeWill::OperatorName operatorName,
        const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
        const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
        const std::map<std::string, std::any> &parameters,
        DataType dataType)
    :m_name(name),
      m_dataType(dataType),
      m_operatorName(operatorName),
      m_inputs(inputs),
      m_outputs(outputs),
      m_parameters(parameters)

{
}

FreeWill::OperatorDescriptor::~OperatorDescriptor()
{

}
