# wfsim makefile

# remove -fopenmp to compile without openmp 
# use /usr/local/bin/g++ in Mac OS X if you installed the alternative compiler for OpenMP http://hpc.sourceforge.net
CC=g++
CFLAGS=-O3 -Wall -fopenmp -I . 

all: wfabc_1 wfabc_2 

install: wfabc_2 wfabc_1
	cp wfabc_2 /usr/local/bin
	cp wfabc_1 /usr/local/bin

wfabc_1: wfsim_stats.cpp
	$(CC) $(CFLAGS) -o wfabc_1 wfsim_stats.cpp

wfabc_2: wfsim.cpp
	$(CC) $(CFLAGS) -o wfabc_2 wfsim.cpp

clean: 
	rm wfabc_1 wfabc_2
