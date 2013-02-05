/* this is just to do some simple c benchmarking */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define SIND(fld, ix, iy, iz) (((fld)->ny*(fld)->nx*(iz)) + \
                               ((fld)->nx*(iy)) + \
                               (ix))
/* interlaced */
#define VIND(fld, ix, iy, iz, ic) (((fld)->ny*(fld)->nx*(fld)->nc*(iz)) + \
                                   ((fld)->nx*(fld)->nc*(iy)) + \
                                   ((fld)->nc*(ix)) + \
                                   (ic))

#define SVAL(fld, ix, iy, iz) ((fld)->dat)[SIND(fld, ix, iy, iz)]
#define VVAL(fld, ix, iy, iz, ic) ((fld)->dat)[VIND(fld, ix, iy, iz, ic)]

typedef double real_t;

typedef struct{
  real_t *dat;
  int nx;
  int ny;
  int nz;
} sfld_t;

typedef struct{
  real_t *dat;
  int nx;
  int ny;
  int nz;
  int nc;
} vfld_t;

void sfld_init(sfld_t *fld, int nx, int ny, int nz){
  fld->dat = malloc(nx * ny * nz * sizeof(real_t));
  fld->nx = nx;
  fld->ny = ny;
  fld->nz = nz;
}

void vfld_init(vfld_t *fld, int nx, int ny, int nz, int nc){
  fld->dat = malloc(nc * nx * ny * nz * sizeof(real_t));
  fld->nx = nx;
  fld->ny = ny;
  fld->nz = nz;
  fld->nc = nc;
}

void sfld_free(sfld_t *fld){
  free(fld->dat);
  fld->dat = NULL;
  fld->nx = 0;
  fld->ny = 0;
  fld->nz = 0;
}

void vfld_free(vfld_t *fld){
  free(fld->dat);
  fld->dat = NULL;
  fld->nx = 0;
  fld->ny = 0;
  fld->nz = 0;
}

double wtime (){
  double value;
  value = (double) clock () / (double) CLOCKS_PER_SEC;
  return value;
}

int main(int argc, char **argv){
  vfld_t v;
  sfld_t mag;
  real_t vx, vy, vz;
  double t0, t1;

  vfld_init(&v, 512, 256, 256, 3);
  sfld_init(&mag, 512, 256, 256);

  for(int k=0; k < v.nz; k++){
    for(int j=0; j < v.ny; j++){
      for(int i=0; i < v.nx; i++){
        VVAL(&v, i, j, k, 0) = (real_t) i*j*k;
        VVAL(&v, i, j, k, 1) = (real_t) -1*i*j*k;
        VVAL(&v, i, j, k, 2) = (real_t) 2*i*j*k;
      }
    }
  }

  t0 = wtime();
  for(int k=0; k < v.nz; k++){
    for(int j=0; j < v.ny; j++){
      for(int i=0; i < v.nx; i++){
// for(int i=0; i < v.nx; i++){
//   for(int j=0; j < v.ny; j++){
//     for(int k=0; k < v.nz; k++){
        vx = VVAL(&v, i, j, k, 0);
        vy = VVAL(&v, i, j, k, 1);
        vz = VVAL(&v, i, j, k, 2);
        SVAL(&mag, i, j, k) = sqrt(vx*vx + vy*vy + vz*vz);
      }
    }
  }

  t1 = wtime();
  printf("Straight C took %lf sec\n", t1 - t0);

  vfld_free(&v);
  sfld_free(&mag);
  return 0;
}

/*
 * EOF
 */
