#ifndef BLOB_H
#define BLOB_H

#include <type_traits>
#include <string>
#include "DeviceSelection.h"

namespace FreeWill 
{
    template<typename DataType = float>
    class Blob_data
    {
    protected:
        const std::string m_name;
        const unsigned int m_batchSize;
        const unsigned int m_depth;
        const unsigned int m_height;
        const unsigned int m_width;
        DataType *m_data;

        Blob_data(const std::string &name, unsigned int batchSize, unsigned int depth, unsigned int height, unsigned int width)
            :m_name(name),
              m_batchSize(batchSize),
              m_depth(depth),
              m_height(height),
              m_width(width)
        {}
    };

	template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Blob : public Blob_data<DataType>
	{

    public:
		Blob(const unsigned int batchSize, 
			 const unsigned int depth,
			 const unsigned int height,
			 const unsigned int width,
			 const std::string &name = "no_name");

        bool init();

		DEVICE_SPECIFIC(CPU | GPU)
		void test();
	};

    template<typename DataType>
    class Blob<GPU, DataType> : public Blob_data<DataType>
    {
    public:
        bool init();

        Blob(const unsigned int batchSize,
             const unsigned int depth,
             const unsigned int height,
             const unsigned int width,
             const std::string &name = "no_name")
            :Blob_data<DataType>(name, batchSize, depth, height, width)
        {}
    };


}

#endif
