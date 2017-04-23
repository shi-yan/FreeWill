#include "RandomNumberGenerator.h"

namespace FreeWill
{


    void RandomNumberGenerator::beginRecording(const std::string &filename)
    {
        if (m_isRecording || m_isReplaying)
        {
            std::cerr << "can't start recording during recording or replaying." << std::endl;
            return;
        }

        m_fileStream = std::fstream();
        m_fileStream.open(filename, std::ios_base::out | std::ios_base::binary);
        m_isRecording = true;
    }

    void RandomNumberGenerator::endRecording()
    {
        if (m_isRecording)
        {
            m_fileStream.seekp (0, m_fileStream.end);
            unsigned int length = m_fileStream.tellp();
            std::cout << length << " bytes, or "<< length / sizeof(float)<<" floats saved for random number recording." << std::endl;

            m_fileStream.close();
            m_isRecording = false;
        }
    }

    void RandomNumberGenerator::beginReplay(const std::string &filename)
    {
        if (m_isRecording || m_isReplaying)
        {
            std::cerr << "can't start replaying during recording or replaying." << std::endl;
            return;
        }

        m_fileStream = std::fstream();
        m_fileStream.open(filename, std::ios_base::in | std::ios_base::binary);
        m_bytesProcessed = 0;
        m_isReplaying = true;
    }

    void RandomNumberGenerator::endReplay()
    {

        if (m_isReplaying)
        {
            m_fileStream.seekg(0, m_fileStream.end);
            unsigned int length = m_fileStream.tellg();
            if (m_bytesProcessed != length)
            {
                std::cout << "random number replay doesn't match recording. recorded: "<<length<<" bytes. Replayed: " <<m_bytesProcessed << " bytes." << std::endl;

            }

            m_fileStream.close();

            m_isReplaying = false;
        }
    }
}
