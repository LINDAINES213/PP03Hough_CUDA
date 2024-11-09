/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 nvcc houghBase.cu -o hough
 ./hough runway.pgm
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common/pgm.h"
#include <cairo/cairo.h>  // Incluir Cairo

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

//*****************************************************************

__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int locID = blockIdx.x * blockDim.x + threadIdx.x; //Definición de locID en base al thread.
  extern __shared__ int localAcc[]; //Acumulador local "localAcc"

  if (locID < (degreeBins * rBins)) {
    localAcc[locID] = 0; //Inicialización a cero de cada elemento del acumulador en memoria compartida.
  }

  __syncthreads(); //Barrera para los hilos del bloque en cuestión.

  //Si el píxel es parte de un borde:
  if (x < w && y < h && pic[y * w + x] > 0) {
    int xCoord = x - w / 2;
    int yCoord = h / 2 - y;

    for (int thetaIdx = 0; thetaIdx < degreeBins; thetaIdx++) {
      float r = xCoord * d_Cos[thetaIdx] + yCoord * d_Sin[thetaIdx];
      int rIdx = (int)((r + rBins / 2.0) / rBins * degreeBins);
      //Modificación del acumulador local.
      atomicAdd(&localAcc[rIdx * degreeBins + thetaIdx], 1); // Ejemplo de suma atómica para el acumulador local
      }
    }
    __syncthreads(); //Segunda barrera.
    
    //Loop al final del kernel para sumar los valores a acc.
    if (locID < (degreeBins * rBins)) {
      atomicAdd(&acc[locID], localAcc[locID]);
    }
  }


//TODO Kernel memoria Constante
__global__ void GPU_HoughTranConst (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  //TODO calcular: int gloID = ?
  //int gloID = w * h + 1; //TODO
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  //TODO calcular: int gloID = ?
  //int gloID = w * h + 1; //TODO
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

void drawHoughLinesCairo (cairo_t *cr, int *hough_acc, float rMax, float rScale, int degreeBins, int rBins, float radInc, int threshold, int w, int h) {
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int acc_val = hough_acc[rIdx * degreeBins + tIdx];
            if (acc_val > threshold) {
                float theta = tIdx * radInc;
                float r = rIdx * rScale - rMax;

                float cosTheta = cos(theta);
                float sinTheta = sin(theta);
                int x0 = r * cosTheta;
                int y0 = r * sinTheta;

                int x1 = static_cast<int>(round(x0 + 1000 * (-sinTheta)));
                int y1 = static_cast<int>(round(y0 + 1000 * (cosTheta)));
                int x2 = static_cast<int>(round(x0 - 1000 * (-sinTheta)));
                int y2 = static_cast<int>(round(y0 - 1000 * (cosTheta)));

                cairo_move_to(cr, x1 + w / 2, y1 + h / 2); // Centrar la imagen
                cairo_line_to(cr, x2 + w / 2, y2 + h / 2); // Centrar la imagen
                cairo_set_source_rgb(cr, 1.0, 0.0, 0.0); // Color rojo
                cairo_set_line_width(cr, 1);
                cairo_stroke(cr);  // Dibujar la línea
            }
        }
    }
}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);

  dim3 blockSize(16, 16);
  dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

  // Tamaño memoria compartida

  size_t sharedMemSize = degreeBins * rBins * sizeof(int);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  GPU_HoughTranShared <<<gridSize, blockSize, sharedMemSize>>> (d_in, w, h, d_hough);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Tiempo de ejecución del kernel: %f ms\n", elapsedTime);

// Dibujar las líneas sobre la imagen original
  cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
  cairo_t *cr = cairo_create(surface);
  // Cargar la imagen original en la superficie de Cairo
  unsigned char *data = inImg.pixels; // Aquí se utilizan los píxeles originales
  for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
          unsigned char pixel = data[y * w + x];
          cairo_set_source_rgb(cr, pixel / 255.0, pixel / 255.0, pixel / 255.0); // Escalar de 0-255 a 0-1
          cairo_rectangle(cr, x, y, 1, 1);
          cairo_fill(cr);
      }
  }

  // Llamar a la función para dibujar las líneas
  drawHoughLinesCairo(cr, h_hough, rMax, rScale, degreeBins, rBins, radInc, 3800, w, h);

  // Guardar la imagen resultante
  cairo_surface_write_to_png(surface, "output.png");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // TODO clean-up
  cudaFree(d_in);
  cudaFree(d_hough);

  free(pcCos);
  free(pcSin);
  free(h_hough);
  free(cpuht);

  return 0;
}
