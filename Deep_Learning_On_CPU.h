#include "constants.h"


class DEEP_LEARNING_ON_CPU {
private:
	double INPUT_WEIGHT[NUM_OF_INPUTS][NUM_OF_NODES_PER_LAYER];
	double INPUT_NET[NUM_OF_NODES_PER_LAYER];
	double INPUT_H[NUM_OF_NODES_PER_LAYER];
	double INPUT_DELTA[NUM_OF_INPUTS][NUM_OF_NODES_PER_LAYER];

#if NUM_OF_HIDDEN_LAYERS > 0
	double HIDDEN_WEIGHT[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER][NUM_OF_NODES_PER_LAYER];
	double HIDDEN_NET[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER];
	double HIDDEN_H[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER];
	double HIDDEN_DELTA[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER][NUM_OF_NODES_PER_LAYER];
#endif

	double OUTPUT_WEIGHT[NUM_OF_NODES_PER_LAYER][NUM_OF_OUTPUTS];
	double OUTPUT_NET[NUM_OF_OUTPUTS];
	double OUTPUT_H[NUM_OF_OUTPUTS];
	double OUTPUT_DELTA[NUM_OF_NODES_PER_LAYER][NUM_OF_OUTPUTS];

	void forward(double *training_point);
	void backward(double *training_point, double *training_target);
	void update();

	void print_DELTA();
	void print_NET();
	void print_H();

public:
	DEEP_LEARNING_ON_CPU() {
		// initialize();
		srand((unsigned)time(NULL));
	}

	void initialize();
	void finalize();
	void training();
	void test();
	void print_WEIGHT();
};



void DEEP_LEARNING_ON_CPU::update() {
	for (int i = 0; i < NUM_OF_INPUTS; ++i)
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			INPUT_WEIGHT[i][j] -= LEARNING_RATE * INPUT_DELTA[i][j];
#if NUM_OF_HIDDEN_LAYERS > 0
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS; ++i)
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			for (int k = 0; k < NUM_OF_NODES_PER_LAYER; ++k)
				HIDDEN_WEIGHT[i][j][k] -= LEARNING_RATE * HIDDEN_DELTA[i][j][k];
#endif
	for (int i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		for (int j = 0; j < NUM_OF_OUTPUTS; ++j)
			OUTPUT_WEIGHT[i][j] -= LEARNING_RATE * OUTPUT_DELTA[i][j];
}

void DEEP_LEARNING_ON_CPU::forward(double *training_point) {
	int i, j, k;

	// Input Layer
	for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
		INPUT_NET[j] = 0;
		for (i = 0; i < NUM_OF_INPUTS; ++i) {
			INPUT_NET[j] += INPUT_WEIGHT[i][j] * training_point[i];
		}
		INPUT_H[j] = ACT_FUNC(INPUT_NET[j]);
	}

#if NUM_OF_HIDDEN_LAYERS > 0
	// First Hidden Layer
	for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
		HIDDEN_NET[0][j] = 0;
		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			HIDDEN_NET[0][j] += HIDDEN_WEIGHT[0][i][j] * INPUT_H[i];
		}
		HIDDEN_H[0][j] = ACT_FUNC(HIDDEN_NET[0][j]);
	}

	// Hidden Layer
	for (k = 1; k < NUM_OF_HIDDEN_LAYERS; ++k)
	{
		for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
			HIDDEN_NET[k][j] = 0;
			for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
				HIDDEN_NET[k][j] += HIDDEN_WEIGHT[k][i][j] * HIDDEN_H[k - 1][i];
			}
			HIDDEN_H[k][j] = ACT_FUNC(HIDDEN_NET[k][j]);
		}
	}

	// Output Layer
	for (j = 0; j < NUM_OF_OUTPUTS; ++j) {
		OUTPUT_NET[j] = 0;
		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			OUTPUT_NET[j] += OUTPUT_WEIGHT[i][j] * HIDDEN_H[NUM_OF_HIDDEN_LAYERS - 1][i];
		}
		OUTPUT_H[j] = ACT_FUNC(OUTPUT_NET[j]);
	}
#else
	// Output Layer
	for (j = 0; j < NUM_OF_OUTPUTS; ++j) {
		OUTPUT_NET[j] = 0;
		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			OUTPUT_NET[j] += OUTPUT_WEIGHT[i][j] * INPUT_H[i];
		}
		OUTPUT_H[j] = ACT_FUNC(OUTPUT_NET[j]);
	}
#endif
}

void DEEP_LEARNING_ON_CPU::backward(double *training_point, double *training_target) {
	int i, j, k;
	double delta[(NUM_OF_OUTPUTS < NUM_OF_NODES_PER_LAYER) ? NUM_OF_NODES_PER_LAYER : NUM_OF_OUTPUTS];
	double temp[(NUM_OF_OUTPUTS < NUM_OF_NODES_PER_LAYER) ? NUM_OF_NODES_PER_LAYER : NUM_OF_OUTPUTS];

#if NUM_OF_HIDDEN_LAYERS == 0
	// Output Layer
	for (j = 0; j < NUM_OF_OUTPUTS; ++j) {
		delta[j] = -(training_target[j] - OUTPUT_H[j]) * delta_ACT_FUNC(OUTPUT_NET[j]);
		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			OUTPUT_DELTA[i][j] += delta[j] * INPUT_H[i];
		}
		temp[j] = delta[j];
	}

	// Input Layer
	for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
		delta[j] = 0;
		for (k = 0; k < NUM_OF_OUTPUTS; ++k) {
			delta[j] += temp[k] * OUTPUT_WEIGHT[j][k];
		}
		delta[j] *= delta_ACT_FUNC(INPUT_NET[j]);

		for (i = 0; i < NUM_OF_INPUTS; ++i) {
			INPUT_DELTA[i][j] += delta[j] * training_point[i];
		}
	}
#elif NUM_OF_HIDDEN_LAYERS > 0
	// Output Layer
	for (j = 0; j < NUM_OF_OUTPUTS; ++j) {
		delta[j] = -(training_target[j] - OUTPUT_H[j]) * delta_ACT_FUNC(OUTPUT_NET[j]);
		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			OUTPUT_DELTA[i][j] += delta[j] * HIDDEN_H[NUM_OF_HIDDEN_LAYERS - 1][i];
		}
		temp[j] = delta[j];
	}


	// Last Hidden Layer
	for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
		delta[j] = 0;
		for (k = 0; k < NUM_OF_OUTPUTS; ++k) {
			delta[j] += temp[k] * OUTPUT_WEIGHT[j][k];
		}
		delta[j] *= delta_ACT_FUNC(HIDDEN_NET[NUM_OF_NODES_PER_LAYER - 1][j]);

		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			HIDDEN_DELTA[NUM_OF_HIDDEN_LAYERS - 1][i][j] += delta[j] * HIDDEN_H[NUM_OF_HIDDEN_LAYERS - 2][i];
		}
	}
	for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		temp[i] = delta[i];


	// Middle of Hidden Layers
	for (int h = NUM_OF_HIDDEN_LAYERS - 2; h > 0; --h)
	{
		for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
			delta[j] = 0;
			for (k = 0; k < NUM_OF_HIDDEN_LAYERS; ++k) {
				delta[j] += temp[k] * HIDDEN_WEIGHT[h + 1][j][k];
			}
			delta[j] *= delta_ACT_FUNC(HIDDEN_NET[h][j]);

			for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
				HIDDEN_DELTA[h][i][j] += delta[j] * HIDDEN_H[h - 1][i];
			}
		}
		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
			temp[i] = delta[i];
	}


	// First Hidden Layer
	for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
		delta[j] = 0;
		for (k = 0; k < NUM_OF_NODES_PER_LAYER; ++k) {
			delta[j] += temp[k] * HIDDEN_WEIGHT[1][j][k];
		}
		delta[j] *= delta_ACT_FUNC(HIDDEN_NET[0][j]);

		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			HIDDEN_DELTA[0][i][j] += delta[j] * INPUT_H[i];
		}
	}
	for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		temp[i] = delta[i];


	// Input Layer
	for (j = 0; j < NUM_OF_NODES_PER_LAYER; ++j) {
		delta[j] = 0;
		for (k = 0; k < NUM_OF_NODES_PER_LAYER; ++k) {
			delta[j] += temp[k] * HIDDEN_WEIGHT[0][j][k];
		}
		delta[j] *= delta_ACT_FUNC(INPUT_NET[j]);

		for (i = 0; i < NUM_OF_INPUTS; ++i) {
			INPUT_DELTA[i][j] += delta[j] * training_point[i];
		}
	}
#endif
}

void DEEP_LEARNING_ON_CPU::training() {
	double error;

	printf("******* Training of NN (Iteration : Error) *******\n");

	for (int epoch = 0, printErr = PRINT_ERROR_PERIOD; epoch < MAX_EPOCH; ++epoch) {
		error = 0;

		memset(INPUT_DELTA, 0, sizeof(double) * NUM_OF_INPUTS * NUM_OF_NODES_PER_LAYER);
		memset(OUTPUT_DELTA, 0, sizeof(double) * NUM_OF_NODES_PER_LAYER * NUM_OF_OUTPUTS);
#if NUM_OF_HIDDEN_LAYERS > 0
		memset(HIDDEN_DELTA, 0, sizeof(double) * NUM_OF_HIDDEN_LAYERS * NUM_OF_NODES_PER_LAYER * NUM_OF_NODES_PER_LAYER);
#endif

		for (int point = 0; point < NUM_OF_TRAINING_DATA; ++point) {
			forward(TRAINING_POINT[point]);

			if (epoch == printErr) {
				//printf("%lf ", error);
				for (int idx = 0; idx < NUM_OF_OUTPUTS; ++idx) {
					error += square(OUTPUT_H[idx] - TRAINING_TARGET[point][idx]);
					//printf("%lf ", square(OUTPUT_H[idx] - TRAINING_TARGET[point][idx]));
				}
				//printf("\n");
			}

			backward(TRAINING_POINT[point], TRAINING_TARGET[point]);
		}
		//print_DELTA();
		update();

		if (epoch == printErr) {
			printf("%d: %lf\n", epoch, error);
			printErr += PRINT_ERROR_PERIOD;
			//print_WEIGHT();
			//print_DELTA();
		}
	}
}

void DEEP_LEARNING_ON_CPU::test() {
	printf("******* Test of NN (Input ; Output of NN) *******\n");

	for (int point = 0; point < NUM_OF_TEST_DATA; ++point) {
		forward(TEST_POINT[point]);

		for (int idx = 0; idx < NUM_OF_INPUTS; ++idx)
			printf("%lf ", TEST_POINT[point][idx]);
		putchar(';');
		putchar(' ');

		for (int idx = 0; idx < NUM_OF_OUTPUTS; ++idx)
			printf("%lf ", OUTPUT_H[idx]);
		putchar('\n');
	}
}


void DEEP_LEARNING_ON_CPU::initialize() {
	for (int i = 0; i < NUM_OF_INPUTS; ++i)
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			INPUT_WEIGHT[i][j] = random_expression;
#if NUM_OF_HIDDEN_LAYERS > 0
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS; ++i)
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			for (int k = 0; k < NUM_OF_NODES_PER_LAYER; ++k)
				HIDDEN_WEIGHT[i][j][k] = random_expression;
#endif
	for (int i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		for (int j = 0; j < NUM_OF_OUTPUTS; ++j)
			OUTPUT_WEIGHT[i][j] = random_expression;
}
void DEEP_LEARNING_ON_CPU::finalize() {
}


void DEEP_LEARNING_ON_CPU::print_NET() {
	printf("**** NET ****\n");
	for (int i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		printf("%lf ", INPUT_NET[i]);
	printf("\n");
#if NUM_OF_HIDDEN_LAYERS > 0
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS; ++i) {
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			printf("%lf ", HIDDEN_NET[i][j]);
		printf("\n");
	}
#endif
	for (int i = 0; i < NUM_OF_OUTPUTS; ++i)
		printf("%lf ", OUTPUT_NET[i]);
	printf("\n");
}
void DEEP_LEARNING_ON_CPU::print_H() {
	printf("***** H *****\n");
	for (int i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		printf("%lf ", INPUT_H[i]);
	printf("\n");
#if NUM_OF_HIDDEN_LAYERS > 0
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS; ++i) {
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			printf("%lf ", HIDDEN_H[i][j]);
		printf("\n");
	}
#endif
	for (int i = 0; i < NUM_OF_OUTPUTS; ++i)
		printf("%lf ", OUTPUT_H[i]);
	printf("\n");
}
void DEEP_LEARNING_ON_CPU::print_DELTA() {
	printf("*** DELTA ***\n");
	for (int i = 0; i < NUM_OF_INPUTS; ++i)
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			printf("%lf ", INPUT_DELTA[i][j]);
	printf("\n");
#if NUM_OF_HIDDEN_LAYERS > 0
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS; ++i) {
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			for (int k = 0; k < NUM_OF_NODES_PER_LAYER; ++k)
				printf("%lf ", HIDDEN_DELTA[i][j][k]);
		printf("\n");
	}
#endif
	for (int i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		for (int j = 0; j < NUM_OF_OUTPUTS; ++j)
			printf("%lf ", OUTPUT_DELTA[i][j]);
	printf("\n");
}
void DEEP_LEARNING_ON_CPU::print_WEIGHT() {
	printf("**** WEIGHT ****\n");
	for (int i = 0; i < NUM_OF_INPUTS; ++i)
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			printf("%lf ", INPUT_WEIGHT[i][j]);
	printf("\n");
#if NUM_OF_HIDDEN_LAYERS > 0
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS; ++i) {
		for (int j = 0; j < NUM_OF_NODES_PER_LAYER; ++j)
			for (int k = 0; k < NUM_OF_NODES_PER_LAYER; ++k)
				printf("%lf ", HIDDEN_WEIGHT[i][j][k]);
		printf("\n");
	}
#endif
	for (int i = 0; i < NUM_OF_NODES_PER_LAYER; ++i)
		for (int j = 0; j < NUM_OF_OUTPUTS; ++j)
			printf("%lf ", OUTPUT_WEIGHT[i][j]);
	printf("\n");
}