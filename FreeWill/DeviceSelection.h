#ifndef DEVICESELECTION_H
#define DEVICESELECTION_H

#define VERIFY_INIT(a) \
    if (! a) \
    {qDebug() << "Operator check failed:" << #a ; abort();}


namespace FreeWill 
{
	typedef enum 
	{
		CPU_NAIVE      = 0x1,
		CPU            = 0x1,
		CPU_SIMD       = 0x2,
		GPU            = 0x4,
		GPU_CUDA       = 0x4
	} DeviceType;
}

#define DEVICE_SPECIFIC(devices) \
    template<DeviceType T = DeviceUsed, typename Enabled = typename std::enable_if<(T & devices) != 0> >

#endif
