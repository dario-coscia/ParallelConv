# Makefile for the ParallelConv.
#
# make verbose   ----> Show the textual representation of the program in execution.
# make nfile     ----> Does not save the blurred image output as PGM image.
# make time      ----> Show the time it took to perform the convolution (one run only, blurred image not saved).
# make debug	 ----> The binary produced will show debug messages.
# make leaks	 ----> Produces an executable which can be used to find find memory leaks.
# make test      ----> Prints only the time of execution (ms)
# make clean	 ----> Clean the folder (removes executable/precompiled headers files)

.DEFAULT_GOAL := output
FLAGS = -std=c++17
UTILITIES = 
ADDITIONAL_FLAGS =
HPC_FLAGS = -O3 -march=native		

# OpenMP or MPI?
ifeq ($(src), mpi)
	SRC = blur_mpi.cpp
	COMPILER = mpic++
	OUT = blur_mpi.x
else ifeq ($(src), openmp)
	SRC = blur_openmp.cpp
	ADDITIONAL_FLAGS += -fopenmp
	COMPILER = g++
	OUT = blur_openmp.x
else ifeq ($(src), hybrid)
	SRC = blur_hybrid.cpp
	ADDITIONAL_FLAGS += -fopenmp
	COMPILER = mpic++
	OUT = blur_hybrid.x
else ifeq ($(src), none)
	SRC = blur_serial.cpp
	COMPILER = g++
	OUT = blur_serial.x
else
	SRC = blur_serial.cpp
	COMPILER = g++
	OUT = blur_serial.x
endif


# floating point number precision
ifeq ($(prec), double)
	ADDITIONAL_FLAGS += -D USE_DOUBLE_PRECISION_DATA
endif



verbose:
	$(COMPILER) $(FLAGS)  $(HPC_FLAGS) $(ADDITIONAL_FLAGS) -g -D VERBOSE -D TIME $(UTILITIES) $(SRC) -o $(OUT)

nfile:
	$(COMPILER) $(FLAGS) $(HPC_FLAGS) $(ADDITIONAL_FLAGS) -D NFILE $(UTILITIES) $(SRC) -o $(OUT)

time:
	$(COMPILER) $(FLAGS) $(HPC_FLAGS) $(ADDITIONAL_FLAGS) -D TIME -D NFILE $(UTILITIES) $(SRC) -o $(OUT)

debug:
	$(COMPILER) $(FLAGS) $(HPC_FLAGS) $(ADDITIONAL_FLAGS) -D NDEBUG $(UTILITIES) $(SRC) -o $(OUT)

leaks:
	$(COMPILER) $(FLAGS) $(ADDITIONAL_FLAGS) -static-libasan -fsanitize=address -g -D VERBOSE -D TIME $(UTILITIES) $(SRC) -o $(OUT)

test:
	$(COMPILER) $(FLAGS)  $(HPC_FLAGS) $(ADDITIONAL_FLAGS) -D TEST -D TIME -D NFILE $(UTILITIES) $(SRC) -o $(OUT)

clean:
	rm *.x
