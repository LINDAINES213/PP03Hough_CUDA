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
#include "common/PGMImage.h"
#include <cairo/cairo.h>

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
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

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

void drawHoughLinesCairo(cairo_t *cr, int* hough_acc, float rMax, float rScale, int degreeBins, int rBins, float radInc, int threshold, int w, int h) {
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int acc_val = hough_acc[rIdx * degreeBins + tIdx];
            if (acc_val > threshold) { // Solo dibujar líneas con alta acumulación
                float theta = tIdx * radInc;
                float r = rIdx * rScale - rMax;

                // Convertir (r, theta) en puntos (x1, y1) y (x2, y2) para dibujar la línea
                float cosTheta = cos(theta);
                float sinTheta = sin(theta);
                int x0 = r * cosTheta;
                int y0 = r * sinTheta;

                // Puntos extremos de la línea para dibujar en la imagen
                int x1 = std::round(x0 + 1000 * (-sinTheta));
                int y1 = std::round(y0 + 1000 * (cosTheta));
                int x2 = std::round(x0 - 1000 * (-sinTheta));
                int y2 = std::round(y0 - 1000 * (cosTheta));

                // Dibujar la línea en la imagen con Cairo
                cairo_move_to(cr, x1, y1);
                cairo_line_to(cr, x2, y2);
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

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

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
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

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

   // Crear una superficie de imagen Cairo
  cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, w, h);
  cairo_t *cr = cairo_create(surface);

  // Dibujar las líneas detectadas usando Cairo
  int threshold = 1500;
  drawHoughLinesCairo(cr, h_hough, rMax, rScale, degreeBins, rBins, radInc, threshold, w, h);

  // Guardar la imagen resultante en formato PNG
  cairo_surface_write_to_png(surface, "imagen_con_lineas.png");

  // Limpiar recursos
  cairo_destroy(cr);
  cairo_surface_destroy(surface);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // TODO clean-up
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);

  free(pcCos);
  free(pcSin);
  free(h_hough);
  free(cpuht);

  return 0;
}
