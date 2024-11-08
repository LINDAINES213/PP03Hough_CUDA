all: pgm.o	houghGlobalConstant

houghGlobal: houghGlobal.cu pgm.o
#nvcc houghBase.cu -o hough `pkg-config --cflags --`
#nvcc houghGlobal.cu pgm.o -o hough -lboost_filesystem -lboost_system `pkg-config --cflags --libs opencv4` -diag-suppress 611
#nvcc houghGlobal.cu pgm.o -o hough \
	-lboost_filesystem -lboost_system \
	-lcairo
	
	nvcc houghGlobal.cu pgm.o -o houghGlobal \
	-lboost_filesystem -lboost_system \
	-lcairo


houghGlobalConstant: houghGlobalConstant.cu pgm.o

	nvcc houghGlobalConstant.cu pgm.o -o houghGlobalConstant \
	-lboost_filesystem -lboost_system \
	-lcairo


pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o

run: houghGlobalConstant
	./houghGlobalConstant runway.pgm