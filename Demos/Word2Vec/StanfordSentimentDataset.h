#ifndef STANFORDSENTIMENTDATASET_H
#define STANFORDSENTIMENTDATASET_H

#include <QString>
#include <QFile>
#include <QHash>
#include "Word2VecDataset.h"
#include <QStringList>
#include <QVector>
#include <map>
#include <cmath>

class StanfordSentimentDataset : public Word2VecDataset
{
private:
    QString m_path;
    unsigned int m_tableSize;
    std::map<std::string, unsigned int> m_tokens;
    std::map<std::string ,unsigned int> m_tokenFrequencies;
    QVector<QStringList> m_sentences;
    QVector<unsigned int> m_sentenceLen;
    unsigned int m_overallSentenceLength;
    QVector<std::string> m_revtokens;
    unsigned int m_wordcount;
    QVector<QStringList> m_allSentencies;
    std::vector<double> m_rejectProb;

public:
    StanfordSentimentDataset(const QString &path, unsigned int tableSize);
    const std::map<std::string, unsigned int> &tokens();
    const QVector<QStringList> &sentences();
    void getRandomContext(unsigned int contextSize, std::string &centerWord, std::vector<std::string> & context);
    unsigned int numSentencies();
    QVector<QStringList> &allSentencies();
    std::vector<double> &rejectProb();
};

#endif // STANFORDSENTIMENTDATASET_H
