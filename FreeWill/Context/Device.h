#ifndef DEVICE_H
#define DEVICE_H

#include "../DeviceSelection.h"

namespace FreeWill
{

    template<DeviceType DeviceUsed>
    class Device{};

    template<>
    class Device<GPU_CUDA>
    {
    
    };
}


#endif
