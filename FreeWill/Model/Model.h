#ifndef MODEL_H
#define MODEL_H
#include <cmath>
#include "../DeviceSelection.h"
#include "../Context/Context.h"
#include <string>
#include <map>
#include <utility>
#include <variant>
#include <any>
#include "TensorDescriptor.h"
#include "OperatorDescriptor.h"


namespace FreeWill
{
    class Model
    {




    private:
        Model();
        Model(const Model &) = delete;
        Model& operator=(const Model &) = delete;

        std::map<std::string, TensorDescriptor*> m_tensors;
        std::map<std::string, OperatorDescriptor*> m_operators;


    public:
        static Model* create();
        ~Model();
        bool init();
        TensorDescriptorHandle addTensor(const std::string &name, const Shape &shape, bool isBatchTensor = true, DataType dataType = FLOAT);
        int addOperator(const std::string &name,
                        const std::string &operatorName,
                        const std::map<std::string, TensorDescriptorHandle> &inputs,
                        const std::map<std::string, TensorDescriptorHandle> &outputs,
                        const std::map<std::string, std::any> &properties = {}, DataType dataType = FLOAT);
        int addOperator(const std::string &name,
                        FreeWill::OperatorName operatorName,
                        const std::map<std::string, TensorDescriptorHandle> &inputs,
                        const std::map<std::string, TensorDescriptorHandle> &outputs,
                        const std::map<std::string, std::any> &properties = {}, DataType DataType = FLOAT);
    };
}

#endif
