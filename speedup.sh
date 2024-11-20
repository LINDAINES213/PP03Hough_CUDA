#!/bin/bash

# Configuración del archivo de salida
output_file="resultados_tiempos_y_eficiencia.csv"

# Encabezados para el archivo CSV
echo "Ejecución,Tiempos_Base_ms,Tiempos_Const_ms,Tiempos_Shared_ms,Speedup_Base_Const,Speedup_Const_Shared,Speedup_Base_Shared" > $output_file #Eficiencia_Base_Const,Eficiencia_Const_Shared,Eficiencia_Base_Shared

# Compilar los programas
echo "Compilando los programas..."
nvcc houghGlobal.cu pgm.o -o houghGlobal \
    -lboost_filesystem -lboost_system -lcairo

nvcc houghGlobalConstant.cu pgm.o -o houghGlobalConstant \
    -lboost_filesystem -lboost_system -lcairo

nvcc houghGlobalConstantShared.cu pgm.o -o houghGlobalConstantShared \
    -lboost_filesystem -lboost_system -lcairo

# Número de ejecuciones
num_runs=20
num_cores=256  # Ajusta al número de cores de tu GPU

# Ejecutar experimentos
echo "Ejecutando experimentos $num_runs veces..."

for ((i=1; i<=num_runs; i++))
do
    echo "Ejecución $i..."

    # Ejecutar houghGlobal y esperar a que termine
    echo "Ejecutando houghGlobal..."
    time_base=$(./houghGlobal runway.pgm | grep "Tiempo de ejecución del kernel" | awk '{print $6}')
    # Esperar que termine completamente antes de continuar
    wait

    # Ejecutar houghGlobalConstant y esperar a que termine
    echo "Ejecutando houghGlobalConstant..."
    time_const=$(./houghGlobalConstant runway.pgm | grep "Tiempo de ejecución del kernel" | awk '{print $6}')
    # Esperar que termine completamente antes de continuar
    wait

    # Ejecutar houghGlobalConstantShared y esperar a que termine
    echo "Ejecutando houghGlobalConstantShared..."
    time_shared=$(./houghGlobalConstantShared runway.pgm | grep "Tiempo de ejecución del kernel" | awk '{print $6}')
    # Esperar que termine completamente antes de continuar
    wait

    # Calcular speedup y eficiencia
    speedup_base_const=$(echo "$time_base / $time_const" | bc -l)
    speedup_const_shared=$(echo "$time_const / $time_shared" | bc -l)
    speedup_base_shared=$(echo "$time_base / $time_shared" | bc -l)

    #efficiency_base_const=$(echo "$speedup_base_const / $num_cores" | bc -l)
    #efficiency_const_shared=$(echo "$speedup_const_shared / $num_cores" | bc -l)
    #efficiency_base_shared=$(echo "$speedup_base_shared / $num_cores" | bc -l)

    # Guardar los resultados de esta ejecución en el archivo CSV 
    echo "$i,$time_base,$time_const,$time_shared,$speedup_base_const,$speedup_const_shared,$speedup_base_shared" >> $output_file #$efficiency_base_const,$efficiency_const_shared,$efficiency_base_shared"

    # Agregar un pequeño retraso para liberar recursos del sistema
    sleep 1 # Este comando es opcional y depende de tus necesidades
done

# Mostrar los resultados finales
echo "Resultados guardados en $output_file"
