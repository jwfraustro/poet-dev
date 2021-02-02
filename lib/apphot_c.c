#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>

#include "utils_c.h"
#include "stats_c.h"
#include "resize_c.h"
#include "disk_c.h"

// Accest to the i,j value of the array a:
#define INDd(a,i,j) *((double *)(a->data + i*a->strides[0] + j*a->strides[1]))

/* make this static if you don't want other code to call this function */
/* I don't make it static because want to access this via ctypes */
/* static */


/* The wrapper to the underlying C function */
static PyObject *apphot_c(PyObject *self, PyObject *args){
  PyArrayObject *image,          *uncert,         *gpmask; // inputs
  //PyArrayObject *image,             *gpmask; // inputs
  //PyObject *imarg,          *unarg,         *gparg; // inputs

  double        **data, **idata, **unc,   **iunc, **mask, **rdata;
  char **imask;

  double  xctr,  yctr,  photap,  skyin,  skyout, skyfrac;    // inputs
  double ixctr, iyctr, iphotap, iskyin, iskyout, iscx, iscy; // expands

  double aplev=0., aperr=0., napix=0., skylev=0., skylevunc=0., nsky=0., 
         nskyideal=0., derrsq=0.;  // outputs

  char **stardisk, **indisk, **outdisk, **idealid, **idealod; // disks

  int *pdstat, *ndisk, *pskystat; // pointers
  double *pslu;                   // pointers

  int diskstatus, ndiskin, ndiskout, ndiskap;                    // utils
  int i, j, nx, ny, mx, my, expand, med, isis, statdummy=1;      // utils
  int xmin, xmax, dx, ymin, ymax, dy;                            // more utils
  int naper=0, nskypix=0, nmask=0, nbad=0, nskyipix=0, status=0; // counters


  if (!PyArg_ParseTuple(args, "OOOddddddii", &image, &uncert, &gpmask,
  //if (!PyArg_ParseTuple(args, "OOddddddii", &image, &gpmask, 
			&xctr, &yctr, &photap, &skyin, &skyout, &skyfrac,
			&expand, &med))
    return NULL;

  //image  = PyArray_FROM_OTF(imarg, NPY_DOUBLE, NPY_IN_ARRAY);
  //uncert = PyArray_FROM_OTF(imarg, NPY_DOUBLE, NPY_IN_ARRAY);
  //gpmask = PyArray_FROM_OTF(imarg, NPY_INT,    NPY_IN_ARRAY);
  // the dimensions of the input 2D array:
  nx = image -> dimensions[0];
  ny = image -> dimensions[1];

  // Work only around the target:
  xmin = MAX(0,  (int)(round(xctr)-skyout-2)); // Left-lower pixels of region
  ymin = MAX(0,  (int)(round(yctr)-skyout-2)); // containing skyout.
  xmax = MIN(nx, (int)(round(xctr)+skyout+2)); // Right-upper pixels of region
  ymax = MIN(ny, (int)(round(yctr)+skyout+2)); // containing skyout.
  dx = xmax - xmin;
  dy = ymax - ymin;

  // Store input data in pointer-to-pointers:
  data = (double **)malloc(dx * sizeof(double *));
  unc  = (double **)malloc(dx * sizeof(double *));
  mask = (double **)malloc(dx * sizeof(double *));
  for (i=0; i<dx; i++){
    *(data+i) = (double *)malloc(dy * sizeof(double));
    *(unc +i) = (double *)malloc(dy * sizeof(double));
    *(mask+i) = (double *)malloc(dy * sizeof(double));
  }
  for   (i = xmin; i < xmax; i++)
    for (j = ymin; j < ymax; j++){
      data[(i-xmin)][(j-ymin)] = INDd(image,  i, j);
      unc [(i-xmin)][(j-ymin)] = INDd(uncert, i, j);
      mask[(i-xmin)][(j-ymin)] = INDd(gpmask, i, j);
    }

  //for(i=0;i<nx;i++) free(image[i]);
  //free(image);

  // Interpolation:
  mx = dx + (dx-1)*(expand-1); // expanded dimensions
  my = dy + (dy-1)*(expand-1);

  idata = (double **)malloc(mx * sizeof(double *));
  iunc  = (double **)malloc(mx * sizeof(double *));
  imask = (char   **)malloc(mx * sizeof(char   *));
  rdata = (double **)malloc(mx * sizeof(double *));
  for (i=0; i<mx; i++){
    *(idata+i) = (double *)malloc(my * sizeof(double));
    *(iunc +i) = (double *)malloc(my * sizeof(double));
    *(imask+i) = (char   *)malloc(my * sizeof(char  ));
    *(rdata+i) = (double *)malloc(my * sizeof(double));
  }
  resize(     idata, data, expand, dx, dy, mx, my); // idata stored in doubles
  resize(      iunc,  unc, expand, dx, dy, mx, my);
  resize_mask(imask, mask, expand, dx, dy, mx, my, rdata); 
  // free malloc'ed memory
  for (i=0; i<dx; i++){
    free(data[i]); free(unc[i]); free(mask[i]);
  }
  free(data); free(unc); free(mask); 
  for (i=0; i<mx; i++)
    free(rdata[i]);
  free(rdata); 

  // expand lengths:
  iphotap = photap      * expand;
  iskyin  = skyin       * expand;
  iskyout = skyout      * expand;
  ixctr   = (xctr-xmin) * expand;
  iyctr   = (yctr-ymin) * expand;

  // make disks:
  indisk   = (char **)malloc(mx * sizeof(char *));
  outdisk  = (char **)malloc(mx * sizeof(char *));
  stardisk = (char **)malloc(mx * sizeof(char *));
  for (i=0; i<mx; i++){
    *(indisk  +i) = (char *)malloc(my * sizeof(char));
    *(outdisk +i) = (char *)malloc(my * sizeof(char));
    *(stardisk+i) = (char *)malloc(my * sizeof(char));
  }

  pdstat = &diskstatus; // disk status pointer
  ndisk = &ndiskin;
  disk(indisk,   iskyin,  ixctr, iyctr, mx, my, pdstat, ndisk);
  ndisk = &ndiskout;
  disk(outdisk,  iskyout, ixctr, iyctr, mx, my, pdstat, ndisk);
  ndisk = &ndiskap;
  disk(stardisk, iphotap, ixctr, iyctr, mx, my, pdstat, ndisk);
  // see status report below

  // allocate useful data in 1D arrays:
  double *skydata;
  double *skyerr;
  skydata = (double *)malloc(sizeof(double)*(ndiskout-ndiskin));
  skyerr  = (double *)malloc(sizeof(double)*(ndiskout-ndiskin));

  for   (i = 0; i < mx; i++)
    for (j = 0; j < my; j++){
      if (outdisk[i][j] && !indisk[i][j]){ // pixels in annulus
        if (!isnan(idata[i][j]) && !isinf(idata[i][j]) && imask[i][j] == 1){
	  skydata[nskypix] = idata[i][j];
	  skyerr [nskypix] = iunc[i][j];
	  nskypix++;
	}
      }
      if (stardisk[i][j]){                 // pixels in aperture
	if (imask[i][j] < 1)
	  nmask++;
	if (isnan(idata[i][j]) || isinf(idata[i][j]))
	  nbad++;
	else{
	  aplev  += idata[i][j];
	  derrsq += iunc[i][j]*iunc[i][j];// sum of data error squared
	  naper++;
	}
      }
    }

  for (i=0; i<mx; i++){
    free(idata[i]);  free(iunc[i]);    free(imask[i]);
    free(indisk[i]); free(outdisk[i]); free(stardisk[i]);
  }
  free(idata); free(iunc); free(imask); 
  free(indisk); free(outdisk); free(stardisk);

  // Re-scaled number of pixels in sky:
  napix = 1.0*naper  /(expand*expand);
  nsky  = 1.0*nskypix/(expand*expand);


  // ideal sky calculation:
  isis = (int)(ceil(iskyout)*2 + 3);            // ideal sky image size
  iscx = fmod(ixctr,1.0) + ceil(iskyout) + 1.0; // ideal sky center x
  iscy = fmod(iyctr,1.0) + ceil(iskyout) + 1.0; // ideal sky center y

  //printf("nx   = %d ", nx);
  //printf("dx   = %d ", dx);
  //printf("mx   = %d ", mx);
  //printf("isis = %d\n", isis);

  // hack
  idealid = (char **)malloc(isis * sizeof(char *));
  idealod = (char **)malloc(isis * sizeof(char *));
  for (i=0; i<isis; i++){
    *(idealid+i) = (char *)malloc(isis * sizeof(char));
    *(idealod+i) = (char *)malloc(isis * sizeof(char));
  }

  ndisk  = &nskyipix;
  pdstat = &statdummy;
  disk(idealid, iskyout, iscx, iscy, isis, isis, pdstat, ndisk);
  nskyideal = (double)nskyipix;  // number of pixels inside inskyout
  disk(idealod,  iskyin, iscx, iscy, isis, isis, pdstat, ndisk);
  nskyideal = (nskyideal - nskyipix) / (expand*expand);

  for(i=0; i<isis; i++){
    free(idealid[i]); free(idealod[i]);
  }
  free(idealid); free(idealod); 
  
  //nskyideal = 10;
  

  // status reports:
  if (nbad > 0)      // NaNs or Inf pixels in aperture
    status |= 1;
  if (naper == 0)    // no good pixels in aperture
    status |= 2;
  if (nmask > 0)     // masked pixels in aperture
    status |= 2;
  if (diskstatus)    // out of bounds aperture
    status |= 4;
  if (nsky < skyfrac * nskyideal)  // sky fraction condition unfulfilled
    status |= 8;
  if (nskypix == 0)  // no good pixels in sky
    status |= 8;


  // sky level calculation:
  pskystat = &statdummy;
  pslu     = &skylevunc;  // pointer to sky level uncertainty
  // mean sky and uncertainty:
  skylev = meanerr(&skydata[0], &skyerr[0], nskypix, pslu, pskystat);
  skylevunc *= expand;
  if (med)  // median sky
    skylev = median(&skydata[0], nskypix);

  free(skydata);
  free(skyerr);

  // photometry
  aplev = (aplev - skylev*naper)/(expand*expand);
  aperr = sqrt(derrsq + naper*skylevunc*skylevunc)/expand;

  // return
  return Py_BuildValue("[d,d,d,d,d,d,d,i]", aplev,  aperr, napix, skylev, 
		       skylevunc, nsky, nskyideal, status);
}

static PyObject *elphot_c(PyObject *self, PyObject *args){
  PyArrayObject *image,          *uncert,         *gpmask; // inputs
  //PyArrayObject *image,             *gpmask; // inputs
  //PyObject *imarg,          *unarg,         *gparg; // inputs

  double        **data, **idata, **unc,   **iunc, **mask, **rdata;
  char **imask;

  double  xctr,  yctr,  xsig,  ysig,  skyin,  skyout, skyfrac, theta; // inputs
  double ixctr, iyctr, ixsig, iysig, iskyin, iskyout, iscx, iscy; // expands

  double aplev=0., aperr=0., napix=0., skylev=0., skylevunc=0., nsky=0., 
         nskyideal=0., derrsq=0.;  // outputs

  char **starell, **indisk, **outdisk, **idealid, **idealod; // disks

  int *pdstat, *ndisk, *pskystat; // pointers
  double *pslu;                   // pointers

  int diskstatus, ndiskin, ndiskout, nellap;                    // utils
  int i, j, nx, ny, mx, my, expand, med, isis, statdummy=1;      // utils
  int xmin, xmax, dx, ymin, ymax, dy;                            // more utils
  int naper=0, nskypix=0, nmask=0, nbad=0, nskyipix=0, status=0; // counters


  if (!PyArg_ParseTuple(args, "OOOddddddddii", &image, &uncert, &gpmask,
  //if (!PyArg_ParseTuple(args, "OOddddddddii", &image, &gpmask, 
			&xctr, &yctr, &xsig, &ysig, &theta,
			&skyin, &skyout, &skyfrac,
			&expand, &med))
    return NULL;

  //image  = PyArray_FROM_OTF(imarg, NPY_DOUBLE, NPY_IN_ARRAY);
  //uncert = PyArray_FROM_OTF(imarg, NPY_DOUBLE, NPY_IN_ARRAY);
  //gpmask = PyArray_FROM_OTF(imarg, NPY_INT,    NPY_IN_ARRAY);
  // the dimensions of the input 2D array:
  nx = image -> dimensions[0];
  ny = image -> dimensions[1];

  // Work only around the target:
  xmin = MAX(0,  (int)(round(xctr)-skyout-2)); // Left-lower pixels of region
  ymin = MAX(0,  (int)(round(yctr)-skyout-2)); // containing skyout.
  xmax = MIN(nx, (int)(round(xctr)+skyout+2)); // Right-upper pixels of region
  ymax = MIN(ny, (int)(round(yctr)+skyout+2)); // containing skyout.
  dx = xmax - xmin;
  dy = ymax - ymin;

  // Store input data in pointer-to-pointers:
  data = (double **)malloc(dx * sizeof(double *));
  unc  = (double **)malloc(dx * sizeof(double *));
  mask = (double **)malloc(dx * sizeof(double *));
  for (i=0; i<dx; i++){
    *(data+i) = (double *)malloc(dy * sizeof(double));
    *(unc +i) = (double *)malloc(dy * sizeof(double));
    *(mask+i) = (double *)malloc(dy * sizeof(double));
  }
  for   (i = xmin; i < xmax; i++)
    for (j = ymin; j < ymax; j++){
      data[(i-xmin)][(j-ymin)] = INDd(image,  i, j);
      unc [(i-xmin)][(j-ymin)] = INDd(uncert, i, j);
      mask[(i-xmin)][(j-ymin)] = INDd(gpmask, i, j);
    }

  // Interpolation:
  mx = dx + (dx-1)*(expand-1); // expanded dimensions
  my = dy + (dy-1)*(expand-1);

  idata = (double **)malloc(mx * sizeof(double *));
  iunc  = (double **)malloc(mx * sizeof(double *));
  imask = (char   **)malloc(mx * sizeof(char   *));
  rdata = (double **)malloc(mx * sizeof(double *));
  for (i=0; i<mx; i++){
    *(idata+i) = (double *)malloc(my * sizeof(double));
    *(iunc +i) = (double *)malloc(my * sizeof(double));
    *(imask+i) = (char   *)malloc(my * sizeof(char  ));
    *(rdata+i) = (double *)malloc(my * sizeof(double));
  }
  resize(     idata, data, expand, dx, dy, mx, my); // idata stored in doubles
  resize(      iunc,  unc, expand, dx, dy, mx, my);
  resize_mask(imask, mask, expand, dx, dy, mx, my, rdata); 
  // free malloc'ed memory
  for (i=0; i<dx; i++){
    free(data[i]); free(unc[i]); free(mask[i]);
  }
  free(data); free(unc); free(mask); 
  for (i=0; i<mx; i++)
    free(rdata[i]);
  free(rdata); 

  // expand lengths:
  ixsig   = xsig        * expand;
  iysig   = ysig        * expand;
  iskyin  = skyin       * expand;
  iskyout = skyout      * expand;
  ixctr   = (xctr-xmin) * expand;
  iyctr   = (yctr-ymin) * expand;

  // make disks:
  indisk   = (char **)malloc(mx * sizeof(char *));
  outdisk  = (char **)malloc(mx * sizeof(char *));
  starell  = (char **)malloc(mx * sizeof(char *));
  for (i=0; i<mx; i++){
    *(indisk +i) = (char *)malloc(my * sizeof(char));
    *(outdisk+i) = (char *)malloc(my * sizeof(char));
    *(starell+i) = (char *)malloc(my * sizeof(char));
  }

  pdstat = &diskstatus; // disk status pointer
  ndisk = &ndiskin;
  disk(indisk,   iskyin,  ixctr, iyctr, mx, my, pdstat, ndisk);
  ndisk = &ndiskout;
  disk(outdisk,  iskyout, ixctr, iyctr, mx, my, pdstat, ndisk);
  ndisk = &nellap;
  ellipse(starell, ixsig, iysig, theta, ixctr, iyctr, mx, my, pdstat, ndisk); 

  // see status report below

  // allocate useful data in 1D arrays:
  double *skydata;
  double *skyerr;
  skydata = (double *)malloc(sizeof(double)*(ndiskout-ndiskin));
  skyerr  = (double *)malloc(sizeof(double)*(ndiskout-ndiskin));
 
  for   (i = 0; i < mx; i++)
    for (j = 0; j < my; j++){
      if (outdisk[i][j] && !indisk[i][j]){ // pixels in annulus
        if (!isnan(idata[i][j]) && !isinf(idata[i][j]) && imask[i][j] == 1){
	  skydata[nskypix] = idata[i][j];
	  skyerr [nskypix] = iunc[i][j];
	  nskypix++;
	}
      }
      if (starell[i][j]){                 // pixels in aperture
	if (imask[i][j] < 1)
	  nmask++;
	if (isnan(idata[i][j]) || isinf(idata[i][j]))
	  nbad++;
	else{
	  aplev  += idata[i][j];
	  derrsq += iunc[i][j]*iunc[i][j];// sum of data error squared
	  naper++;
	}
      }
    }

  for (i=0; i<mx; i++){
    free(idata[i]);  free(iunc[i]);    free(imask[i]);
    free(indisk[i]); free(outdisk[i]); free(starell[i]);
  }
  free(idata); free(iunc); free(imask); 
  free(indisk); free(outdisk); free(starell);

  // Re-scaled number of pixels in sky:
  napix = 1.0*naper  /(expand*expand);
  nsky  = 1.0*nskypix/(expand*expand);


  // ideal sky calculation:
  isis = (int)(ceil(iskyout)*2 + 3);            // ideal sky image size
  iscx = fmod(ixctr,1.0) + ceil(iskyout) + 1.0; // ideal sky center x
  iscy = fmod(iyctr,1.0) + ceil(iskyout) + 1.0; // ideal sky center y

  // hack
  idealid = (char **)malloc(isis * sizeof(char *));
  idealod = (char **)malloc(isis * sizeof(char *));
  for (i=0; i<isis; i++){
    *(idealid+i) = (char *)malloc(isis * sizeof(char));
    *(idealod+i) = (char *)malloc(isis * sizeof(char));
  }

  ndisk  = &nskyipix;
  pdstat = &statdummy;
  disk(idealid, iskyout, iscx, iscy, isis, isis, pdstat, ndisk);
  nskyideal = (double)nskyipix;  // number of pixels inside inskyout
  disk(idealod,  iskyin, iscx, iscy, isis, isis, pdstat, ndisk);
  nskyideal = (nskyideal - nskyipix) / (expand*expand);

  for(i=0; i<isis; i++){
    free(idealid[i]); free(idealod[i]);
  }
  free(idealid); free(idealod); 

  //nskyideal = 10;
  

  // status reports:
  if (nbad > 0)      // NaNs or Inf pixels in aperture
    status |= 1;
  if (naper == 0)    // no good pixels in aperture
    status |= 2;
  if (nmask > 0)     // masked pixels in aperture
    status |= 2;
  if (diskstatus)    // out of bounds aperture
    status |= 4;
  if (nsky < skyfrac * nskyideal)  // sky fraction condition unfulfilled
    status |= 8;
  if (nskypix == 0)  // no good pixels in sky
    status |= 8;


  // sky level calculation:
  pskystat = &statdummy;
  pslu     = &skylevunc;  // pointer to sky level uncertainty
  // mean sky and uncertainty:
  skylev = meanerr(&skydata[0], &skyerr[0], nskypix, pslu, pskystat);
  skylevunc *= expand;
  if (med)  // median sky
    skylev = median(&skydata[0], nskypix);

  free(skydata);
  free(skyerr);

  // photometry
  aplev = (aplev - skylev*naper)/(expand*expand);
  aperr = sqrt(derrsq + naper*skylevunc*skylevunc)/expand;

  // return
  return Py_BuildValue("[d,d,d,d,d,d,d,i]", aplev,  aperr, napix, skylev, 
		       skylevunc, nsky, nskyideal, status);
}

/* The module doc string */
PyDoc_STRVAR(apphot_c_module__doc__,
  "Gaussian point evalutation kernel");

/* The function doc strings */
PyDoc_STRVAR(apphot_c__doc__,
  "Circular-interpolated aperture photometry");
PyDoc_STRVAR(elphot_c__doc__,
  "Elliptical-interpolated aperture photometry");

/* A list of all the methods defined by this module. */
static PyMethodDef apphot_c_methods[] = {
    {"apphot_c", apphot_c, METH_VARARGS, apphot_c__doc__},
    {"elphot_c", elphot_c, METH_VARARGS, elphot_c__doc__},
    {NULL,       NULL,     0,            NULL}      /* sentinel */
};

static struct PyModuleDef apphot_c_module =
{
    PyModuleDef_HEAD_INIT,
    "apphot_c",             // name of module 
    apphot_c_module__doc__, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    apphot_c_methods
};

/* When Python imports a C module named 'X' it loads the module */
/* then looks for a method named "PyInit_"+X and calls it.         */
PyMODINIT_FUNC PyInit_apphot_c(void)
{
    import_array();
    return PyModule_Create(&apphot_c_module);
}

/* old from python 2
void initapphot_c(){
  Py_InitModule("apphot_c", apphot_c_methods);
  import_array();
}
*/
