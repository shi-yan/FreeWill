#include "endian.h"
#include "Tensor/Tensor.h"
#include <cstdio>

int main()
{
    FILE *fp = fopen("train-images-idx3-ubyte","rb");
    FILE *lfp = fopen("train-labels-idx1-ubyte","rb");

    unsigned int magicNum = 0;
    unsigned int numOfImage = 0;
    unsigned int numOfRow = 0;
    unsigned int numOfColumn = 0;

    unsigned int magicNumLabel = 0;
    unsigned int labelCount = 0;
    fread(&magicNumLabel, sizeof(unsigned int), 1, lfp);
    fread(&labelCount, sizeof(unsigned int ),1,lfp);

    magicNumLabel = be32toh(magicNumLabel);
    labelCount = be32toh(labelCount);

    fread(&magicNum, sizeof(unsigned int), 1, fp);
    fread(&numOfImage, sizeof(unsigned int), 1, fp);
    fread(&numOfRow, sizeof(unsigned int), 1, fp);
    fread(&numOfColumn, sizeof(unsigned int), 1,fp);

    magicNum = be32toh(magicNum);
    numOfImage = be32toh(numOfImage);
    numOfRow = be32toh(numOfRow);
    numOfColumn = be32toh(numOfColumn);

    printf("magic number: %x\n", magicNum);
    printf("num of image: %d\n", numOfImage);
    printf("num of row: %d\n", numOfRow);
    printf("num of column: %d\n", numOfColumn);

    printf("magic number label: %x\n", magicNumLabel);
    printf("num of label: %d\n", labelCount);

    for(unsigned int image = 0;image<3;++image)
    {
        for(unsigned int y = 0 ; y < numOfRow; ++y)
        {
            for(unsigned int x = 0;x< numOfColumn; ++x)
            {
                unsigned char pixel = 0;
                fread(&pixel, sizeof(unsigned char), 1, fp);
                printf("%3d,", pixel);
            }
            printf("\n");
        }
        unsigned char label = 0;
        fread(&label, sizeof(unsigned char), 1, lfp);
        printf("lable: %d\n", label);
    }
    fclose(lfp);
    fclose(fp);
    
    return 0;
}
