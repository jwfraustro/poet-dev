#include <math.h>

void disk(char **disk, double radius, double xctr, double yctr, 
	    int nx, int ny, int *status, int *ndisk){
  // make a pointer to pointers of doubles to emulate a matrix
  int i, j, n=0;
  //matrix_char(disk, dvec, nx, ny);

  // alert if the center lies outside the image:
  if ( (xctr-radius) < 0 || (xctr+radius) > (nx-1) ||
       (yctr-radius) < 0 || (yctr+radius) > (ny-1) ){
    *status = 1;
  }
  else
    *status = 0;

  for   (i=0; i<nx; i++)
    for (j=0; j<ny; j++){
      // is the point disk[i][j] inside the disk?
      disk[i][j] = (i-xctr)*(i-xctr) + (j-yctr)*(j-yctr) <= radius*radius;
      n += disk[i][j];
    }

  // set the number of pixels within radius in ndisk
  *ndisk = n;

  return;
}

void ellipse(char **ellipse, double xsig, double ysig, double theta,
	     double xctr, double yctr, 
	     int nx, int ny, int *status, int *nellipse){
  /*
  This function makes an elliptical maks with given parameters. 
  Used by the elphot_c function in apphot_c.c. This function
  treats indexing as (y,x).

  Perplexingly, apphot_c treats indexing as (x,y) and functions
  perfectly fine, despite being used in the same way as this function. 
  This is probably due to symmetry, both in distance
  to the edges and rotationally, which is not the case in elphot_c.
  */
  
  // make a pointer to pointers of doubles to emulate a matrix
  int i, j, n=0;
  //matrix_char(disk, dvec, nx, ny);

  double maxd; // maximum distance from center

  maxd = fmax(xsig, ysig);

  // alert if the center lies outside the image:
  if ( (xctr-maxd*cos(theta)) < 0 || (xctr+maxd*cos(theta)) > (nx-1) ||
       (yctr-maxd*sin(theta)) < 0 || (yctr+maxd*sin(theta)) > (ny-1) ){
    *status = 1;
  }
  else
    *status = 0;

  for   (i=0; i<nx; i++)
    for (j=0; j<ny; j++){
      // is the point ellipse[j][i] inside the ellipse?
      ellipse[i][j] = ysig*ysig*pow( cos(theta)*(i-xctr)-sin(theta)*(j-yctr),2) 
	            + xsig*xsig*pow(-sin(theta)*(i-xctr)-cos(theta)*(j-yctr),2) 
	            <= ysig*ysig*xsig*xsig;
      n += ellipse[i][j];
    }

  // set the number of pixels in nellipse
  *nellipse = n;

  return;
}
