NVCC      = nvcc
ARCH      = sm_75  # [ADDED] T4 arch. Change to sm_80 for A100, sm_86 for A10/A30 on Lambda
CFLAGS    = -O2 -std=c++14
INCLUDES  = -Isrc

SRC_DIR   = src
SRCS      = $(SRC_DIR)/main.cu $(SRC_DIR)/nbody.cu
BENCH_SRC = $(SRC_DIR)/benchmark.cu $(SRC_DIR)/nbody.cu

all: main benchmark

main: $(SRCS)
	$(NVCC) -arch=$(ARCH) $(CFLAGS) $(INCLUDES) $(SRCS) -o main

benchmark: $(BENCH_SRC)
	$(NVCC) -arch=$(ARCH) $(CFLAGS) $(INCLUDES) $(BENCH_SRC) -o benchmark

clean:
	rm -f main benchmark
	rm -rf output/

run: main
	mkdir -p output
	./main $(N) $(STEPS) $(DT) $(CENTRIPETAL)

.PHONY: all clean run
