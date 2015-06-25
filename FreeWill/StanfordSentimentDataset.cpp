#include "StanfordSentimentDataset.h"
#include <QDebug>
#include <time.h>

StanfordSentimentDataset::StanfordSentimentDataset(const QString &path, unsigned int tableSize)
    :m_path(path),
      m_tableSize(tableSize),
      m_tokens(),
      m_tokenFrequencies(),
      m_sentences(),
      m_sentenceLen()
{

}

const std::map<std::string, unsigned int>  &StanfordSentimentDataset::tokens()
{
    if (m_tokens.size())
    {
        return m_tokens;
    }

    m_wordcount = 0;

    unsigned int idx = 0;

    sentences();

    for(int i = 0;i<m_sentences.size();++i)
    {
        for(int w = 0;w<m_sentences[i].size();++w)
        {
            m_wordcount ++;
            if (m_tokens.find(m_sentences[i].at(w).toStdString()) == m_tokens.end() )
            {
                m_tokens[m_sentences[i].at(w).toStdString()] = idx++;
                m_revtokens.push_back(m_sentences[i].at(w).toStdString());
                m_tokenFrequencies[m_sentences[i].at(w).toStdString()] = 1;
            }
            else
            {
                m_tokenFrequencies[m_sentences[i].at(w).toStdString()] += 1;
            }
        }
    }

    m_tokens["UNK"] = idx;
    m_revtokens.push_back("UNK");
    m_tokenFrequencies["UNK"] = 1;
    m_wordcount += 1;

    return m_tokens;
}

const QVector<QStringList>& StanfordSentimentDataset::sentences()
{
    if (m_sentences.size())
    {
        return m_sentences;
    }

    QFile sentencesFile(QString("%1/datasetSentences.txt").arg(m_path));

    sentencesFile.open(QFile::ReadOnly);

    bool firstLine = true;

    while(!sentencesFile.atEnd())
    {
        QByteArray line = sentencesFile.readLine();

        if (firstLine)
        {
            //skip the first line as it has only format info.
            firstLine = false;
            continue;
        }

        QString lineInString = QString::fromUtf8(line).trimmed();

        //qDebug() << lineInString;

        QStringList splitted = lineInString.split(QRegExp("\\t+|\\s+"), QString::SkipEmptyParts);

        splitted.removeFirst();

        m_sentences.push_back(splitted);
    }

    sentencesFile.close();

    m_sentenceLen.resize(m_sentences.size());
    m_overallSentenceLength = 0;

    for(int i = 0; i<m_sentenceLen.size();++i)
    {
        m_sentenceLen[i] = m_sentences[i].size();
        m_overallSentenceLength += m_sentenceLen[i];
    }

    return m_sentences;
}

unsigned int StanfordSentimentDataset::numSentencies()
{
    sentences();
    return m_sentences.size();
}


std::vector<double> &StanfordSentimentDataset::rejectProb()
{
    if (m_rejectProb.size())
    {
        return m_rejectProb;
    }

    tokens();

    double threshold = 1e-5 * m_wordcount;

    m_rejectProb.resize(m_tokens.size(), 0.0);

    for(int i = 0; i< m_tokens.size(); ++i)
    {
        std::string w = m_revtokens[i];
        double freq = 1.0 * m_tokenFrequencies[w];
        m_rejectProb[i] = std::max(0.0, 1.0 - std::sqrt(threshold / freq));
    }

    return m_rejectProb;
}

QVector<QStringList> &StanfordSentimentDataset::allSentencies()
{
    if (m_allSentencies.size())
    {
        return m_allSentencies;
    }

    sentences();
    rejectProb();
    tokens();

    srand(time(NULL));

    for(int i = 0; i<30;++i)
    {
        for(int e = 0;e<m_sentences.size();++e)
        {

            QStringList newSent;
            for(int w = 0;w<m_sentences[e].size();++w)
            {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                if (m_rejectProb[ m_tokens[m_sentences[e][w].toStdString()]] <= 0 || r >= m_rejectProb[ m_tokens[m_sentences[e][w].toStdString()]])
                {
                    newSent.append(m_sentences[e][w]);
                }

            }

            if (newSent.size() > 1)
            {
                m_allSentencies.push_back(newSent);
            }
        }
    }
}

void StanfordSentimentDataset::getRandomContext(unsigned int contextSize, std::string &centerWord, std::vector<std::string> & context)
{
    allSentencies();

    unsigned int sentenceId = rand() % m_allSentencies.size();
    QStringList &sent = m_allSentencies[sentenceId];

    unsigned int wordId = rand() % sent.size();

    for(int i = std::max(0, (int)wordId - (int)contextSize); i<wordId ; ++i)
    {
        if (sent[wordId] != sent[i])
        {
            context.push_back(sent[i].toStdString());
        }
    }

    centerWord = sent[wordId].toStdString();

    if (wordId + 1 < sent.size())
    {
        unsigned int till = std::min(sent.size(), (int)wordId + (int)contextSize + 1);
        for(int i = wordId + 1; i < till ; ++i)
        {
            if (sent[wordId] != sent[i])
            {
                context.push_back(sent[i].toStdString());
            }
        }
    }
}
