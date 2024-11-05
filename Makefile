all: pgm.o	hough

hough:	houghGlobal.cu pgm.o
#nvcc houghBase.cu -o hough `pkg-config --cflags --`
	nvcc houghGlobal.cu pgm.o -o hough -lboost_filesystem -lboost_system `pkg-config --cflags --libs opencv4` -diag-suppress 611
pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o

run: hough
	./hough runway.pgm