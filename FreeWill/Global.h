#ifndef GLOBAL_H
#define GLOBAL_H

#define BLOCK_SIZE 32

typedef enum
{
    Sigmoid,
    Rectifier,
    Tanh,
    None
} Activation;

typedef enum
{
    CrossEntropy,
    MeanSquared
} CostFunction;

#endif // GLOBAL_H
