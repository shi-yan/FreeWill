#include "Blob/Blob.h"

namespace FreeWill
{
	template<>
	template<DeviceType, typename>
	void Blob<CPU, float>::test()
	{
	}

    template<typename DataType>
    bool Blob<GPU, DataType>::init(){}
}
