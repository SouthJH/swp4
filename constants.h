#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <math.h>


#define		NUM_OF_INPUTS				2	// The number of inputs
#define		NUM_OF_OUTPUTS				1	// The number of outputs
#define		NUM_OF_HIDDEN_LAYERS		5	// The number of hidden layers
#define		NUM_OF_NODES_PER_LAYER		10	// The number of nodes in the hidden layer

#define		MAX_EPOCH					10000000
#define		LEARNING_RATE				0.5

#define		PRINT_ERROR_PERIOD			100000

#define		NUM_OF_TRAINING_DATA		4
#define		NUM_OF_TEST_DATA			4


double TRAINING_POINT[NUM_OF_TRAINING_DATA][NUM_OF_INPUTS] = {
	{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
};
double TRAINING_TARGET[NUM_OF_TRAINING_DATA][NUM_OF_OUTPUTS] = {
	{0.0}, {1.0}, {1.0}, {0.0}
};
double TEST_POINT[NUM_OF_TEST_DATA][NUM_OF_INPUTS] = {
	{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
};



#if 0
#define		LReLU(x)				((x > 0)? x : 0.01 * x)
#define		ACT_FUNC(x)				((x > 0)? x : 0.01 * x)
#define		delta_ACT_FUNC(x)		((x > 0)? 1 : 0.01)

#elif 0
#define		ReLU(x)					((x > 0)? x : 0)
#define		ACT_FUNC(x)				((x > 0)? x : 0)
#define		delta_ACT_FUNC(x)		((x > 0)? 1 : 0)

#elif 1
#define		SIGMOID(x)				(1./(1+exp(-(x))))
#define		ACT_FUNC(x)				SIGMOID(x)
#define		delta_ACT_FUNC(x)		(SIGMOID(x) * (1 - SIGMOID(x)))
#endif

#define		random_expression			rand()/((double)RAND_MAX * 2)
#define		square(a)					((a)*(a))
#define		max(a, b)					((a > b) ? a : b)
//#define		max(a, b, c)				((a > b && a > c) ? a : ((b > a && b > c) ? b : c))