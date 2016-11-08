
namespace FreeWill 
{
	typedef enum 
	{
		CPU_NAIVE      = 0x1,
		CPU            = 0x2,
		CPU_SIMD       = 0x2,
		GPU            = 0x4,
		GPU_CUDA       = 0x4
	} DeviceType;
}