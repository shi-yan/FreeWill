#ifndef SOLVER_H
#define SOLVER_H

#include "../DeviceSelection.h"

namespace FreeWill
{
    class Solver
    {
    public:
        DeviceType m_deviceUsed;
        unsigned int m_batchSize;
    };
}

#endif
