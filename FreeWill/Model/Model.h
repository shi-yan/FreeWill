#ifndef MODEL_H
#define MODEL_H
#include <cmath>
#include "../DeviceSelection.h"
#include "../Context/Context.h"
#include <string>
#include "../Tensor/Tensor.h"
#include <map>
#include <utility>
#include "../Tensor/Shape.h"
#include <variant>
namespace FreeWill
{
    class Model
    {
    private:
        typedef enum
        {
            FLOAT,
            DOUBLE,
            UNSIGNED_INT
        } DataType;

        class TensorDescriptor
        {
            friend class Model;
        public:
            std::string m_name;
            Shape m_shape;
            bool m_isBatchTensor;
            int m_batchSize;
            DataType m_dataType;
            //change this to variant or any
            std::map<DeviceType, std::variant<TensorBase<GPU_CUDA> *>> m_tensors;

            TensorDescriptor(const std::string &name, const Shape &shape, DataType dataType = FLOAT, bool isBatchTensor = false);
            ~TensorDescriptor();

            void operator=(const TensorDescriptor &in);
            TensorDescriptor(const TensorDescriptor &in);
        };

        class OperatorDescriptor
        {
            friend class Model;
        private:
            std::string m_name;
            DataType m_dataType;
            
            OperatorDescriptor(const std::string &name);
            ~OperatorDescriptor();
        };


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
        int addTensor(const std::string &name, const Shape &shape, bool isBatchTensor = true);
    };
}

#endif
