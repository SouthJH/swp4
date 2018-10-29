#include "Deep_Learning_On_CPU.h"

int main() {
	DEEP_LEARING_ON_CPU nn;

	nn.initialize();
	nn.training();

	nn.test();

	return 0;
}