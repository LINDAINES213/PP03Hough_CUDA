# 🚀 Transformada de Hough en CUDA

Este proyecto implementa la **Transformada Lineal de Hough** en CUDA, un algoritmo esencial en el procesamiento de imágenes para la detección de líneas rectas.

## 📋 Descripción del Proyecto

Se implementa una técnica que ayuda a detectar líneas rectas en una imagen en blanco y negro con un modelo orientado en CUDA, identificando qué píxeles pertenecen a una línea específica. Cada píxel "iluminado" vota por posibles líneas a las que podría pertenecer, y las líneas con más votos son seleccionadas. En este proyecto, aplicamos el modelo CUDA para paralelizar este proceso y optimizar el tiempo de ejecución utilizando diferentes tipos de memoria.

## 🎯 Objetivos

1. **Entender y aplicar memoria Constante en CUDA**.
2. **Aprovechar las memorias Global, Compartida y Constante** en un problema común de análisis de imágenes.
3. **Implementar y optimizar la Transformada de Hough** para detectar líneas en imágenes de alta precisión.

## 🧩 Estructura del Código

- **houghBase.cu**: Contiene el núcleo CUDA que ejecuta la Transformada de Hough.
- **config.h**: Define parámetros de configuración.
- **pgm.cpp y pgm.h**: Funciones para cargar y guardar imágenes en formato PGM.
- **Makefile**: Para compilar y ejecutar el proyecto.

## 🛠️ Instrucciones de Uso

**Compilar el Proyecto**
   ```bash
   make ./hough <imagen.pgm> <output.pgm>
   ```

## 🚀 Funcionalidades Clave

### Implementación Paralela: 
El algoritmo está diseñado para ejecutarse en múltiples hilos, mejorando la velocidad de procesamiento de imágenes grandes.

### Optimización de Memoria:
- Memoria Global: Almacena la imagen y la matriz acumuladora de votos.
- Memoria Constante: Almacena valores trigonométricos de ángulos para reducir el costo computacional.
- Memoria Compartida: Acumulador local de votos para cada bloque, sincronizando y reduciendo accesos a la memoria global.

## 📈 Medición de Rendimiento
Usamos eventos de CUDA para medir el tiempo de ejecución en las versiones con:

- Solo memoria global
- Memoria global + Constante
- Memoria global + Constante + Compartida

## 👥 Autores
Diego Alexander Hernández Silvestre, 21270 🎓
Mario Antonio Guerra Morales, 21008 🖥️
Linda Inés Jiménez Vides, 21169 🔍
