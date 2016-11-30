#ifndef OPERATOR_H
#define OPERATOR_H

#include <cstring>
#include "../Tensor/ReferenceCountedBlob.h"
#include "../DeviceSelection.h"
#include <map>


namespace FreeWill
{
    template <DeviceType DeviceUsed = CPU>
    class Operator
    {
    public:
        Operator(){};

        virtual void evaluate() = 0;
        virtual bool init() = 0;
       
      
        virtual ~Operator(){};
    };
}
#endif
