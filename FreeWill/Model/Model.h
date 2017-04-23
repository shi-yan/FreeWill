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


namespace FreeWill
{
    class Solver;
    class Model
    {
        friend class Solver;

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
        TensorDescriptorHandle addTensor(const std::string &name, const Shape &shape, bool isBatchTensor = false, bool isRandomlyInitialized = true, DataType dataType = DataType::FLOAT);
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
        const DataType *readonlyAccess(const TensorDescriptorHandle &tensorDescriptorHandle)
        {
           TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.first];

           TensorBase<DeviceUsed>* tensorBase = std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][0]);

           return static_cast<const DataType*>(tensorBase->cpuDataHandle());
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
        DataType *beginMutateData(const TensorDescriptorHandle &tensorDescriptorHandle)
        {
           TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.first];

           TensorBase<DeviceUsed>* tensorBase = std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][0]);

           return static_cast<DataType*>(tensorBase->cpuDataHandle());
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
        void endMutateData(const TensorDescriptorHandle &tensorDescriptorHandle)
        {
            TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.first];

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

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
        void clearTensor(const TensorDescriptorHandle &tensorDescriptorHandle)
        {
            TensorDescriptor* tensorDescriptor = m_tensors[tensorDescriptorHandle.first];

            for(unsigned int i = 1; i < tensorDescriptor->m_tensors[DeviceUsed].size(); ++i)
            {
                std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][i])->clear();

                if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
                {
                    std::get<TensorBase<DeviceUsed>*>(tensorDescriptor->m_tensors[DeviceUsed][i])->copyFromHostToDevice();
                }
            }

        }


    };
}

#endif
