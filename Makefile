CPP = nvcc -arch=compute_35 -code=sm_35 -Wno-deprecated-gpu-targets -Xcompiler "-Ofast  -fopenmp"

SRCS = main.cu fluid_solver.cu EventManager.cpp


all: phase2

phase2:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

runseq .ONESHELL:
	export OMP_NUM_THREADS=1
	
	./fluid_sim

runpar:
	export OMP_NUM_THREADS=20
	./fluid_sim
