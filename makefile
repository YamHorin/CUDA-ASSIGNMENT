finale:
	nvcc -gencode arch=compute_61,code=sm_61 scan_strcmp.cu -o foo 
clean:
	rm -f foo