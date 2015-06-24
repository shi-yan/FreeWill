#ifndef WORD2VECDATASET_H
#define WORD2VECDATASET_H

#include <string>
#include <vector>

class Word2VecDataset
{
public:
    Word2VecDataset();

    virtual void getRandomContext(unsigned int contextSize, std::string &centerWord, std::vector<std::string> & context) const = 0;
};

#endif // WORD2VECDATASET_H
