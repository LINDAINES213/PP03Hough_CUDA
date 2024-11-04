all: pgm.o	hough

hough:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough -lboost_filesystem -lboost_system

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
