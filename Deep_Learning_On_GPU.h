#include "constants.h"

#ifdef __INTELLISENSE__
void __syncthreads();
double atomicAdd(double* address, double val);
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define		ERROR_CHECK(err) \
	if (err != cudaSuccess) { \
		printf("[%s:%d] ERROR %d\n", __FILE__, __LINE__, err); \
		exit(EXIT_FAILURE); \
	}

class DEEP_LEARNING_ON_GPU {
private:
	double INPUT_WEIGHT[NUM_OF_INPUTS][NUM_OF_NODES_PER_LAYER], *input_delta_on_gpu;
	__device__ double INPUT_WEIGHT_ON_GPU[NUM_OF_INPUTS][NUM_OF_NODES_PER_LAYER];
	__device__ double INPUT_NET_ON_GPU[NUM_OF_NODES_PER_LAYER];
	__device__ double INPUT_H_ON_GPU[NUM_OF_NODES_PER_LAYER];
	__device__ double INPUT_DELTA_ON_GPU[NUM_OF_INPUTS][NUM_OF_NODES_PER_LAYER];

#if NUM_OF_HIDDEN_LAYERS > 0
	double HIDDEN_WEIGHT[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER][NUM_OF_NODES_PER_LAYER], *hidden_delta_on_gpu;
	__device__ double HIDDEN_WEIGHT_ON_GPU[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER][NUM_OF_NODES_PER_LAYER];
	__device__ double HIDDEN_NET_ON_GPU[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER];
	__device__ double HIDDEN_H_ON_GPU[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER];
	__device__ double HIDDEN_DELTA_ON_GPU[NUM_OF_HIDDEN_LAYERS][NUM_OF_NODES_PER_LAYER][NUM_OF_NODES_PER_LAYER];
#endif

	double OUTPUT_WEIGHT[NUM_OF_NODES_PER_LAYER][NUM_OF_OUTPUTS], *output_delta_on_gpu;
	double OUTPUT_H[NUM_OF_OUTPUTS];
	__device__ double OUTPUT_WEIGHT_ON_GPU[NUM_OF_NODES_PER_LAYER][NUM_OF_OUTPUTS];
	__device__ double OUTPUT_NET_ON_GPU[NUM_OF_OUTPUTS];
	__device__ double OUTPUT_H_ON_GPU[NUM_OF_OUTPUTS];
	__device__ double OUTPUT_DELTA_ON_GPU[NUM_OF_NODES_PER_LAYER][NUM_OF_OUTPUTS];

	double *TRAINING_POINT_ON_GPU;
	double *TRAINING_TARGET_ON_GPU;


	__device__ void reduction_input(int inp, int dst, double *training_point);
#if NUM_OF_HIDDEN_LAYERS > 0
	__device__ void reduction_hidden(int inp, int dst, int index, double *training_point);
#endif
	__device__ void reduction_output(int inp, int dst, double *training_point);
	__device__ void forward(int inp, int dst, double *training_point);
	__device__ void backward(int inp, int dst, double *training_point, double *training_target);
	__global__ void update();
	__global__ void training_kernel(double *training_point, double *training_target);

	__host__ void write_WEIGHT();
	__host__ void read_WEIGHT();

public:
	DEEP_LEARNING_ON_GPU() {
		cudaError_t err;
		srand((unsigned)time(NULL));
		err = cudaMalloc((void **)&TRAINING_POINT_ON_GPU, sizeof(double) * NUM_OF_INPUTS);		ERROR_CHECK(err);
		err = cudaMalloc((void **)&TRAINING_TARGET_ON_GPU, sizeof(double) * NUM_OF_OUTPUTS);	ERROR_CHECK(err);

		err = cudaGetSymbolAddress((void **)&input_delta_on_gpu, "INPUT_DELTA_ON_GPU");			ERROR_CHECK(err);
		err = cudaGetSymbolAddress((void **)&output_delta_on_gpu, "OUTPUT_DELTA_ON_GPU");		ERROR_CHECK(err);
#if NUM_OF_HIDDEN_LAYERS > 0
		err = cudaGetSymbolAddress((void **)&hidden_delta_on_gpu, "HIDDEN_DELTA_ON_GPU");		ERROR_CHECK(err);
#endif
	}

	__host__ void initialize();
	__host__ void finalize();
	__host__ void training();
	__host__ void test();
	__host__ void print_WEIGHT();
};


__global__ void DEEP_LEARNING_ON_GPU::update() {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int jdx = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < NUM_OF_INPUTS && jdx < NUM_OF_NODES_PER_LAYER)
		INPUT_WEIGHT[idx][jdx] -= LEARNING_RATE * INPUT_DELTA_ON_GPU[idx][jdx];
#if NUM_OF_HIDDEN_LAYERS > 0
	for (int layer = 0; layer < NUM_OF_HIDDEN_LAYERS; ++layer)
		if (idx < NUM_OF_NODES_PER_LAYER && jdx < NUM_OF_NODES_PER_LAYER)
			HIDDEN_WEIGHT[layer][idx][jdx] -= LEARNING_RATE * HIDDEN_DELTA_ON_GPU[layer][idx][jdx];
#endif
	if (idx < NUM_OF_NODES_PER_LAYER && jdx < NUM_OF_OUTPUTS)
		OUTPUT_WEIGHT[idx][jdx] -= LEARNING_RATE * OUTPUT_DELTA_ON_GPU[idx][jdx];
}

__device__ void DEEP_LEARNING_ON_GPU::reduction_input(int inp, int dst, double *training_point)
{
	int y = threadIdx.y;

	__shared__ double temp[NUM_OF_NODES_PER_LAYER];

	if (inp == 0)
		INPUT_NET_ON_GPU[dst] = 0;
	__syncthreads();

	temp[y] = (y < NUM_OF_NODES_PER_LAYER) ? INPUT_WEIGHT_ON_GPU[inp][dst] * training_point[inp] : 0;
	__syncthreads();

	for (int p = dst / 2; p >= 1; p = p >> 1) {
		if (y < p)
			temp[y] += temp[y + p];
		__syncthreads();
	}

	if (y == 0) {
		// INPUT_NET_ON_GPU[dst] += temp[0];
		atomicAdd(&INPUT_NET_ON_GPU[dst], temp[0]);
	}
	__syncthreads();

	if (inp == 0)
		INPUT_H_ON_GPU[dst] = ACT_FUNC(INPUT_NET_ON_GPU[dst]);
	__syncthreads();
}
__device__ void DEEP_LEARNING_ON_GPU::reduction_output(int inp, int dst, double *training_point) {
	for (j = 0; j < NUM_OF_OUTPUTS; ++j) {
		OUTPUT_NET[j] = 0;
		for (i = 0; i < NUM_OF_NODES_PER_LAYER; ++i) {
			OUTPUT_NET[j] += OUTPUT_WEIGHT[i][j] * HIDDEN_H[NUM_OF_HIDDEN_LAYERS - 1][i];
		}
		OUTPUT_H[j] = ACT_FUNC(OUTPUT_NET[j]);
	}

	int y = threadIdx.y;

	__shared__ double temp[NUM_OF_NODES_PER_LAYER];

	if (inp == 0)
		INPUT_NET_ON_GPU[dst] = 0;
	__syncthreads();

	temp[y] = (y < NUM_OF_NODES_PER_LAYER) ? INPUT_WEIGHT_ON_GPU[inp][dst] * training_point[inp] : 0;
	__syncthreads();

	for (int p = dst / 2; p >= 1; p = p >> 1) {
		if (y < p)
			temp[y] += temp[y + p];
		__syncthreads();
	}

	if (y == 0) {
		// INPUT_NET_ON_GPU[dst] += temp[0];
		atomicAdd(&INPUT_NET_ON_GPU[dst], temp[0]);
	}
	__syncthreads();

	if (inp == 0)
		INPUT_H_ON_GPU[dst] = ACT_FUNC(INPUT_NET_ON_GPU[dst]);
	__syncthreads();
}
#if NUM_OF_HIDDEN_LAYERS > 0
__device__ void DEEP_LEARNING_ON_GPU::reduction_hidden(int inp, int dst, int index, double *training_point) {
	int y = threadIdx.y;

	__shared__ double temp[NUM_OF_NODES_PER_LAYER];

	if (inp == 0)
		HIDDEN_NET_ON_GPU[index][dst] = 0;
	__syncthreads();

	if (index == 0)
		temp[y] = (y < NUM_OF_NODES_PER_LAYER) ? HIDDEN_WEIGHT_ON_GPU[index][inp][dst] * INPUT_H_ON_GPU[inp] : 0;
	else if (index > 0)
		temp[y] = (y < NUM_OF_NODES_PER_LAYER) ? HIDDEN_WEIGHT_ON_GPU[index][inp][dst] * HIDDEN_H_ON_GPU[index - 1][inp] : 0;
	else
		temp[y] = 0;
	__syncthreads();

	for (int p = dst / 2; p >= 1; p = p >> 1) {
		if (y < p)
			temp[y] += temp[y + p];
		__syncthreads();
	}

	if (y == 0) {
		// INPUT_NET_ON_GPU[dst] += temp[0];
		atomicAdd(&HIDDEN_NET_ON_GPU[index][dst], temp[0]);
	}
	__syncthreads();

	if (inp == 0)
		HIDDEN_H_ON_GPU[index][dst] = ACT_FUNC(HIDDEN_NET_ON_GPU[index][dst]);
	__syncthreads();
}
#endif
__device__ void DEEP_LEARNING_ON_GPU::forward(int inp, int dst, double *training_point) {
	// Input Layer
	if (inp < NUM_OF_INPUTS && dst < NUM_OF_NODES_PER_LAYER) {
		// INPUT_NET_ON_GPU[dst] += INPUT_WEIGHT_ON_GPU[inp][dst] * training_point[inp];
		reduction_input(inp, dst, training_point);
	}


#if NUM_OF_HIDDEN_LAYERS > 0
	// Hidden Layer
	 if (inp < NUM_OF_NODES_PER_LAYER && dst < NUM_OF_NODES_PER_LAYER) {
		 for (int k = 0; k < NUM_OF_HIDDEN_LAYERS; ++k) {
			 reduction_hidden(inp, dst, k, training_point);
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

__device__ void DEEP_LEARNING_ON_GPU::backward(int inp, int dst, double *training_point, double *training_target) {
}

__global__ void DEEP_LEARNING_ON_GPU::training_kernel(double *training_point, double *training_target) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int jdx = blockIdx.y * blockDim.y + threadIdx.y;

	forward(idx, jdx, training_point);
	backward(idx, jdx, training_point, training_target);
}

__host__ void DEEP_LEARNING_ON_GPU::training() {
	double error;
	cudaError_t err;

	dim3 threads(16, 16);
	dim3 blocks(max(NUM_OF_INPUTS, NUM_OF_NODES_PER_LAYER) / threads.x, 
		max(NUM_OF_NODES_PER_LAYER, NUM_OF_OUTPUTS) / threads.y);

	printf("******* Training of NN (Iteration : Error) *******\n");

	for (int epoch = 0, printErr = PRINT_ERROR_PERIOD; epoch < MAX_EPOCH; ++epoch) {
		error = 0;

		err = cudaMemsetAsync(input_delta_on_gpu, 0, sizeof(double) * NUM_OF_INPUTS * NUM_OF_NODES_PER_LAYER);								ERROR_CHECK(err);
		err = cudaMemsetAsync(output_delta_on_gpu, 0, sizeof(double) * NUM_OF_NODES_PER_LAYER * NUM_OF_OUTPUTS);							ERROR_CHECK(err);
#if NUM_OF_HIDDEN_LAYERS > 0
		err = cudaMemset(hidden_delta_on_gpu, 0, sizeof(double) * NUM_OF_HIDDEN_LAYERS * NUM_OF_NODES_PER_LAYER * NUM_OF_NODES_PER_LAYER);	ERROR_CHECK(err);
#endif

		for (int point = 0; point < NUM_OF_TRAINING_DATA; ++point) {
			err = cudaMemcpyAsync(TRAINING_POINT_ON_GPU, TRAINING_POINT[point], sizeof(double) * NUM_OF_INPUTS, cudaMemcpyHostToDevice);	ERROR_CHECK(err);
			err = cudaMemcpy(TRAINING_TARGET_ON_GPU, TRAINING_TARGET[point], sizeof(double) * NUM_OF_OUTPUTS, cudaMemcpyHostToDevice);		ERROR_CHECK(err);

			training_kernel <<< blocks, threads >>> (TRAINING_POINT_ON_GPU, TRAINING_TARGET_ON_GPU);
			err = cudaDeviceSynchronize();	ERROR_CHECK(err);

			if (epoch == printErr) {
				// read output_h fromsymbol
				err = cudaMemcpy(OUTPUT_H, OUTPUT_H_ON_GPU, sizeof(double) * NUM_OF_OUTPUTS, cudaMemcpyDeviceToHost);
				ERROR_CHECK(err);
				for (int idx = 0; idx < NUM_OF_OUTPUTS; ++idx) {
					error += square(OUTPUT_H[idx] - TRAINING_TARGET[point][idx]);
				}
			}
		}
		
		update <<< blocks, threads >>> ();
		err = cudaDeviceSynchronize();	ERROR_CHECK(err);

		if (epoch == printErr) {
			printf("%d: %lf\n", epoch, error);
			printErr += PRINT_ERROR_PERIOD;
		}
	}
}

__host__ void DEEP_LEARNING_ON_GPU::test() {
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
__host__ void DEEP_LEARNING_ON_GPU::initialize() {
	cudaError_t err;
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
	write_WEIGHT();

	//err = cudaMemcpy(TRAINING_POINT_ON_GPU, TRAINING_POINT, sizeof(double) * NUM_OF_TRAINING_DATA * NUM_OF_INPUTS, cudaMemcpyHostToDevice);		ERROR_CHECK(err);
	//err = cudaMemcpy(TRAINING_TARGET_ON_GPU, TRAINING_TARGET, sizeof(double) * NUM_OF_TRAINING_DATA * NUM_OF_OUTPUTS, cudaMemcpyHostToDevice);	ERROR_CHECK(err);
}
__host__ void DEEP_LEARNING_ON_GPU::finalize() {
	cudaFree(TRAINING_POINT_ON_GPU);
	cudaFree(TRAINING_TARGET_ON_GPU);
	cudaFree(input_delta_on_gpu);
	cudaFree(output_delta_on_gpu);
#if NUM_OF_HIDDEN_LAYERS > 0
	cudaFree(hidden_delta_on_gpu);
#endif
}
__host__ void DEEP_LEARNING_ON_GPU::write_WEIGHT() {
	cudaError_t err;
	err = cudaMemcpyToSymbol(INPUT_WEIGHT_ON_GPU, INPUT_WEIGHT, sizeof(double) * NUM_OF_INPUTS * NUM_OF_NODES_PER_LAYER);		ERROR_CHECK(err);
#if NUM_OF_HIDDEN_LAYERS > 0
	err = cudaMemcpyToSymbol(HIDDEN_WEIGHT_ON_GPU, HIDDEN_WEIGHT, sizeof(double) * NUM_OF_HIDDEN_LAYERS * NUM_OF_NODES_PER_LAYER * NUM_OF_NODES_PER_LAYER);		ERROR_CHECK(err);
#endif
	err = cudaMemcpyToSymbol(OUTPUT_WEIGHT_ON_GPU, OUTPUT_WEIGHT, sizeof(double) * NUM_OF_NODES_PER_LAYER * NUM_OF_OUTPUTS);	ERROR_CHECK(err);
}
__host__ void DEEP_LEARNING_ON_GPU::read_WEIGHT() {
	cudaError_t err;
	err = cudaMemcpyFromSymbol(INPUT_WEIGHT, INPUT_WEIGHT_ON_GPU, sizeof(double) * NUM_OF_INPUTS * NUM_OF_NODES_PER_LAYER);		ERROR_CHECK(err);
#if NUM_OF_HIDDEN_LAYERS > 0
	err = cudaMemcpyFromSymbol(HIDDEN_WEIGHT, HIDDEN_WEIGHT_ON_GPU, sizeof(double) * NUM_OF_HIDDEN_LAYERS * NUM_OF_NODES_PER_LAYER * NUM_OF_NODES_PER_LAYER);	ERROR_CHECK(err);
#endif
	err = cudaMemcpyFromSymbol(OUTPUT_WEIGHT, OUTPUT_WEIGHT_ON_GPU, sizeof(double) * NUM_OF_NODES_PER_LAYER * NUM_OF_OUTPUTS);	ERROR_CHECK(err);
}
__host__ void DEEP_LEARNING_ON_GPU::print_WEIGHT() {
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