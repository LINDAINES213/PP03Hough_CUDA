all: pgm.o	houghGlobalConstant

houghGlobal: houghGlobal.cu pgm.o
	
	nvcc houghGlobal.cu pgm.o -o houghGlobal \
	-lboost_filesystem -lboost_system \
	-lcairo


houghGlobalConstant: houghGlobalConstant.cu pgm.o

	nvcc houghGlobalConstant.cu pgm.o -o houghGlobalConstant \
	-lboost_filesystem -lboost_system \
	-lcairo


houghGlobalConstantShared: houghGlobalConstantShared.cu pgm.o

	nvcc houghGlobalConstantShared.cu pgm.o -o houghGlobalConstantShared \
	-lboost_filesystem -lboost_system \
	-lcairo

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o

run: houghGlobalConstant
	./houghGlobalConstant runway.pgm