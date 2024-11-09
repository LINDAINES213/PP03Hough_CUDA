#!/bin/bash

# Compilar ambos programas
echo "Compilando ambos programas..."
nvcc houghGlobal.cu pgm.o -o houghGlobal \
-lboost_filesystem -lboost_system \
-lcairo
nvcc houghGlobalConstant.cu pgm.o -o houghGlobalConstant \
-lboost_filesystem -lboost_system \
-lcairo

# Ejecutar los programas y capturar el tiempo
echo "Ejecutando hough_base..."
time_base=$(./houghGlobal runway.pgm | grep "Tiempo de ejecución del kernel" | awk '{print $6}')

echo "Ejecutando hough_const..."
time_const=$(./houghGlobalConstant runway.pgm | grep "Tiempo de ejecución del kernel" | awk '{print $6}')

# Imprimir los tiempos obtenidos
echo "Tiempo de ejecución hough_base: $time_base ms"
echo "Tiempo de ejecución hough_const: $time_const ms"

# Calcular speedup y eficiencia
speedup=$(echo "$time_base / $time_const" | bc -l)
num_cores=256  # Sustituye por el número de cores que tiene tu GPU si es diferente
efficiency=$(echo "$speedup / $num_cores" | bc -l)

# Mostrar los resultados
echo "Speedup: $speedup"
echo "Efficiency: $efficiency"