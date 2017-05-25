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
#include "Solver.h"
#include <sstream>


namespace FreeWill
{
    class Solver;
    class TensorDescriptorHandle;
    class Model
    {
        friend class Solver;
        friend class TensorDescriptorHandle;

    private:
        Model();
        Model(const Model &) = delete;
        Model& operator=(const Model &) = delete;

        std::map<std::string, TensorDescriptor*> m_tensors;
        std::map<std::string, OperatorDescriptor*> m_operators;
        std::vector<std::pair<TensorDescriptorHandle, TensorDescriptorHandle>> m_updatePairs;

        std::vector<OperatorDescriptorHandle> m_forwardPath;
        std::vector<OperatorDescriptorHandle> m_backwardPath;



    public:
        static Model* create();
        ~Model();
        bool init(Solver const &solver);
        TensorDescriptorHandle addTensor(const std::string &name, const Shape &shape, DataType dataType = DataType::FLOAT, bool isBatchTensor = false, bool isRandomlyInitialized = false);
        OperatorDescriptorHandle addOperator(const std::string &name,
                        const std::string &operatorName,
                        const std::map<std::string, TensorDescriptorHandle> &inputs,
                        const std::map<std::string, TensorDescriptorHandle> &outputs,
                        const std::map<std::string, std::any> &properties = {}, DataType dataType = DataType::FLOAT);
        OperatorDescriptorHandle addOperator(const std::string &name,
                        FreeWill::OperatorName operatorName,
                        const std::map<std::string, TensorDescriptorHandle> &inputs,
                        const std::map<std::string, TensorDescriptorHandle> &outputs,
                        const std::map<std::string, std::any> &properties = {}, DataType DataType = DataType::FLOAT);

        void generateSVGDiagram(const std::string &filename);

        bool defineForwardPath(const std::vector<OperatorDescriptorHandle> &forwardOperators);

        bool defineBackwardPath(const std::vector<OperatorDescriptorHandle> &backwardOperators);

        bool defineWeightUpdatePairs(const std::vector<std::pair<TensorDescriptorHandle, TensorDescriptorHandle>> &updatePairs);

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
        const DataType *readonlyAccess(const TensorDescriptorHandle &tensorDescriptorHandle, int deviceId = 0)
        {
           TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.name()];

           TensorBase<DeviceUsed>* tensorBase = std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][deviceId]);

           return static_cast<const DataType*>(tensorBase->cpuDataHandle());
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
        DataType *beginMutateData(const TensorDescriptorHandle &tensorDescriptorHandle, int deviceId = 0)
        {
           TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.name()];

           TensorBase<DeviceUsed>* tensorBase = std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][deviceId]);

           return static_cast<DataType*>(tensorBase->cpuDataHandle());
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
        void endMutateData(const TensorDescriptorHandle &tensorDescriptorHandle, int deviceId = -1)
        {
            TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.name()];

            if (deviceId < 0)
            {
                unsigned char *sourcePtr = static_cast<unsigned char*>(std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][0])->cpuDataHandle());
                unsigned int sourceSize = std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][0])->sizeInByte();

                for(unsigned int i = 1; i < tensorDescriptor->m_tensors[DeviceUsed].size(); ++i)
                {
                    unsigned char *destPtr = static_cast<unsigned char*>(std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][i])->cpuDataHandle());

                    std::copy(sourcePtr, sourcePtr + sourceSize, destPtr);

                    if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
                    {
                        std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][i])->copyFromHostToDevice();
                    }
                }
            }
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
        void clearTensor(const TensorDescriptorHandle &tensorDescriptorHandle)
        {
            TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.name()];

            for(unsigned int i = 0; i < tensorDescriptor->m_tensors[DeviceUsed].size(); ++i)
            {
                std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][i])->clear();

                if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
                {
                    std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][i])->copyFromHostToDevice();
                }
            }

        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
        std::string debugOutputTensor(const TensorDescriptorHandle &TensorDescriptorHandle)
        {
            TensorDescriptor* tensorDescriptor = m_tensors[TensorDescriptorHandle.name()];

            std::stringstream ss;

            for(unsigned int i = 0; i < tensorDescriptor->m_tensors[DeviceUsed].size(); ++i)
            {
                ss << (*std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][i])->template toType<DataType>()) << std::endl;
            }

            return ss.str();
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
        void generateGradientMergeOperators(std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> &operatorList,
                                            const TensorDescriptorHandle &tensorDescriptorHandle)
        {

            TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.name()];


            TensorBase<DeviceUsed> *tensorBase = std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][0]);

            for(unsigned int i = 1; i < tensorDescriptor->m_tensors[DeviceUsed].size(); ++i)
            {
                TensorBase<DeviceUsed> *tensor = std::get<TensorBase<DeviceUsed>*> (tensorDescriptor->m_tensors[DeviceUsed][i]);

                ElementwiseAdd<DeviceUsed, DataType> *elementwiseAdd = new ElementwiseAdd<DeviceUsed, DataType>();
                elementwiseAdd->setInputParameter("OperandA", tensorBase);
                elementwiseAdd->setInputParameter("OperandB", tensor);
                elementwiseAdd->setOutputParameter("Result", tensorBase);
                elementwiseAdd->init();

                operatorList.push_back(elementwiseAdd);
            }

        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
        void generateUpdateFirstDeviceTensorOperators(std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> &operatorList,
                                                      const TensorDescriptorHandle &tensorDescriptorHandle,
                                                      const TensorDescriptorHandle &gradientDescriptorHandle)
        {
            TensorDescriptor *operandATensorDescriptor = m_tensors[tensorDescriptorHandle.name()];

            TensorDescriptor *operandBTensorDescriptor = m_tensors[gradientDescriptorHandle.name()];

            TensorBase<DeviceUsed> *operandATensorBase = std::get<TensorBase<DeviceUsed>*>(operandATensorDescriptor->m_tensors[DeviceUsed][0]);
            TensorBase<DeviceUsed> *operandBTensorBase = std::get<TensorBase<DeviceUsed>*>(operandBTensorDescriptor->m_tensors[DeviceUsed][0]);

            ElementwiseAdd<DeviceUsed, DataType> *elementwiseAdd = new ElementwiseAdd<DeviceUsed, DataType>();
            elementwiseAdd->setInputParameter("OperandA", operandATensorBase);
            elementwiseAdd->setInputParameter("OperandB", operandBTensorBase);
            elementwiseAdd->setOutputParameter("Result", operandATensorBase);
            elementwiseAdd->init();
            operatorList.push_back(elementwiseAdd);
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
        void generateBroadcastFirstDeviceTensorOperators(std::vector<std::variant<Operator<DeviceType::CPU_NAIVE>*, Operator<DeviceType::GPU_CUDA>*>> &operatorList,
                                                      const TensorDescriptorHandle &tensorDescriptorHandle)
        {
            TensorDescriptor *operandATensorDescriptor = m_tensors[tensorDescriptorHandle.name()];

            TensorBase<DeviceUsed> *operandATensorBase = std::get<TensorBase<DeviceUsed>*>(operandATensorDescriptor->m_tensors[DeviceUsed][0]);

            for(unsigned int i = 1; i < operandATensorDescriptor->m_tensors[DeviceUsed].size(); ++i)
            {
                TensorBase<DeviceUsed> *operandATensorBaseDup = std::get<TensorBase<DeviceUsed>*>(operandATensorDescriptor->m_tensors[DeviceUsed][i]);

                ElementwiseAdd<DeviceUsed, DataType> *elementwiseAdd = new ElementwiseAdd<DeviceUsed, DataType>(0.0f);

                elementwiseAdd->setInputParameter("OperandA", operandATensorBase);
                elementwiseAdd->setInputParameter("OperandB", operandATensorBase);
                elementwiseAdd->setOutputParameter("Result", operandATensorBaseDup);

                operatorList.push_back(elementwiseAdd);


            }
        }

    };
}

#endif
