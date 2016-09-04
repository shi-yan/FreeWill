#ifndef FREEWILL_H
#define FREEWILL_H

#include "Global.h"

#define __ON_CPU__ template < typename std::enable_if<!UseGpu>::type>
#define __ON_GPU__ template < typename std::enable_if<UseGpu>::type>

#define __USE_CPU__ false
#define __USE_GPU__ true

#endif // FREEWILL_H
