#ifndef RANDOMNUMBERGENERATOR_H
#define RANDOMNUMBERGENERATOR_H

#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <random>

namespace FreeWill
{
    class RandomNumberGenerator
    {
        std::fstream m_fileStream;
        unsigned int m_bytesProcessed;

        RandomNumberGenerator()
            :m_fileStream(),
              m_bytesProcessed(0),
              m_isRecording(false),
              m_isReplaying(false)
        {}
        ~RandomNumberGenerator(){}

        bool m_isRecording;
        bool m_isReplaying;

        int m_debug;

    public:
        static RandomNumberGenerator &getSingleton()
        {
            static RandomNumberGenerator obj;
            return obj;
        }

        template<typename DataType = float>
        DataType getRandom()
        {
            DataType value = 0;

            if (m_isReplaying)
            {
                m_fileStream.read((char*)&value, sizeof(value));

                m_bytesProcessed += sizeof(value);

                return value;
            }

            static std::mt19937 gen(std::time(NULL));
            std::normal_distribution<DataType> normDis(0, 1);

            value = normDis(gen);

            //std::cerr << "rand:" << m_debug++ << std::endl;

            if (m_isRecording)
            {
                m_fileStream.write((char*)&value, sizeof(value));
            }

            return value;
        }

        void beginRecording(const std::string &filename);
        void endRecording();

        void beginReplay(const std::string &filename);
        void endReplay();
    };


}

#endif
