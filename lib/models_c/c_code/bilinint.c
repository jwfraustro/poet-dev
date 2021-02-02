#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))
#define IND_int(a,i) *((int *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))
#define IND2(a,i,j) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]+j*PyArray_STRIDES(a)[1]))
#define IND2_int(a,i,j) *((int *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]+j*PyArray_STRIDES(a)[1]))

static PyObject *bilinint(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *bilinint(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *posflux, *retbinflux, *retbinstd, *issmoothing;
  PyObject *wbfipmask;
  PyArrayObject *x, *y, *flux, *binfluxmask, *kernel, *binloc, *dydx, *etc, *ipparams;
  //need to make some temp tuples to read in from argument list, then parse
  // a,a,a,a,a dtype=int,a,tuple[d,d,d,d],array[a,a]dtyp=int,array[a,a,a,a],tuple[int,int],bool
  PyObject *tup1, *tup2;
  
  //initialize the keywords
  retbinflux = Py_False;
  retbinstd  = Py_False;

  //make the keywords list
  static char *kwlist[] = {"ipparams","posflux","etc","retbinflux","retbinstd",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|OOO",kwlist,&ipparams,&posflux\
                                  ,&etc,&retbinflux,&retbinstd))
    {
      return NULL;
    }


  //now we must break appart the posflux tuple
  y           = (PyArrayObject *) PyList_GetItem(posflux,0);
  x           = (PyArrayObject *) PyList_GetItem(posflux,1);
  flux        = (PyArrayObject *) PyList_GetItem(posflux,2);
  wbfipmask   =                   PyList_GetItem(posflux,3);
  binfluxmask = (PyArrayObject *) PyList_GetItem(posflux,4);
  kernel      = (PyArrayObject *) PyList_GetItem(posflux,5);
  tup1        =                   PyList_GetItem(posflux,6);
  binloc      = (PyArrayObject *) PyList_GetItem(posflux,7);
  dydx        = (PyArrayObject *) PyList_GetItem(posflux,8);
  tup2        =                   PyList_GetItem(posflux,9);
  issmoothing =                   PyList_GetItem(posflux,10);

  //create the arrays the will be returned, under various conditions
  PyArrayObject *output, *binflux, *binstd, *tempwbfip;
  npy_intp dims[1];

  dims[0] = PyArray_DIMS(flux)[0];
  output  = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  dims[0] = PyList_Size(wbfipmask);
  binflux = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  binstd  = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  int dis = PyArray_DIMS(binfluxmask)[0];
  int i,j,arsize,temp_int,counter;
  double temp_mean,temp_std,meanbinflux;

  //need to make a lock to deal with the menbinflux variable
  //omp_lock_t lck;
  //omp_init_lock(&lck);

  counter = 0;
  meanbinflux = 0;
  //remind keving to make all wbfipmask things arrays
  //#pragma omp parallel for shared(lck,meanbinflux,counter) private(j,tempwbfip, \
  //                                arsize,temp_mean,temp_std,temp_int)
  for(i = 0; i<dis;i++)
    {
      if(IND_int(binfluxmask,i) == 1)
        {
          if(PyObject_IsTrue(retbinstd) == 1)
            {
              tempwbfip = (PyArrayObject *) PyList_GetItem(wbfipmask,i);
	      arsize = PyArray_DIMS(tempwbfip)[0];
              temp_mean = 0;
              temp_std  = 0;
              for(j=0;j<arsize;j++)
                {
                  temp_int   = IND_int(tempwbfip,j); 
                  temp_mean += (IND(flux,temp_int)/IND(etc,temp_int));
                }
              temp_mean /= (double) arsize;
              
              for(j=0;j<arsize;j++)
                {
                  temp_int  = IND_int(tempwbfip,j);
                  temp_std += pow(((IND(flux,temp_int)/IND(etc,temp_int))\
                                   -temp_mean),2);
                }
              temp_std /= (double) arsize;
              temp_std = sqrt(temp_std);

              IND(binflux,i) = temp_mean;
              IND(binstd,i)  = temp_std;
              
              //omp_set_lock(&lck);
              meanbinflux += temp_mean;
              counter += 1;
              //omp_unset_lock(&lck);
            }
          else
            {
              tempwbfip = (PyArrayObject *) PyList_GetItem(wbfipmask,i);
	      arsize = PyArray_DIMS(tempwbfip)[0];
              temp_mean = 0;
              for(j=0;j<arsize;j++)
                {
                  temp_int   = IND_int(tempwbfip,j);
                  temp_mean += (IND(flux,temp_int)/IND(etc,temp_int));
                }
              temp_mean /= (double) arsize;
              IND(binflux,i) = temp_mean;

              //omp_set_lock(&lck);
              meanbinflux += temp_mean;
              counter     += 1;
              //omp_unset_lock(&lck);
            }
        }
      else
        {
          IND(binflux,i) = 0;
          IND(binstd, i) = 0;
        }
    }
  meanbinflux /= (double) counter;

  //#pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      IND(binflux,i) /= meanbinflux;
      IND(binstd, i) /= meanbinflux;
    }
  

  PyObject *thing1, *thing2, *reshape1, *reshape2, *arglist, *dict;
  PyArrayObject *temparr;
  if(PyCallable_Check(issmoothing))
    {
      thing1 = PyTuple_New(2);
      thing2 = PyTuple_New(2);
      PyTuple_SetItem(thing1,0,PyTuple_GetItem(tup1,0));
      PyTuple_SetItem(thing1,0,PyTuple_GetItem(tup1,1));
      PyTuple_SetItem(thing2,0,PyTuple_GetItem(tup1,2));
      PyTuple_SetItem(thing2,1,PyTuple_GetItem(tup1,3));
      reshape1 = PyArray_Reshape(binflux,tup2);
      reshape2 = PyArray_Reshape(binfluxmask,tup2);
      arglist  = Py_BuildValue("OOOO",reshape1,thing1,thing2,reshape2);
      dict     = Py_BuildValue("{s:O}","gk",kernel);
      temparr  = (PyArrayObject *) PyObject_Call(issmoothing,arglist,dict);
      Py_XDECREF(binflux);
      binflux  = (PyArrayObject *) PyArray_Flatten(temparr, NPY_CORDER);
      Py_XDECREF(temparr);
      Py_XDECREF(thing1);
      Py_XDECREF(thing2);
      Py_XDECREF(reshape1);
      Py_XDECREF(reshape2);
      Py_XDECREF(arglist);
      Py_XDECREF(dict);
    }
  

  //int xsize,t_int_one,t_int_x,t_int_x_one;
  int t_int_one;
  int xsize =  (int) PyLong_AS_LONG(PyTuple_GetItem(tup2,1));
  int t_int_x, t_int_x_one;
  dims[0] = PyArray_DIMS(binloc)[1];

  //#pragma omp parallel for private(temp_int,t_int_one,t_int_x,t_int_x_one)
  for(i=0;i<dims[0];i++)
    {
      temp_int = IND2_int(binloc,1,i);
      t_int_one = temp_int +1;
      t_int_x   = temp_int+xsize;
      t_int_x_one = t_int_x +1;
      IND(output,i) = IND(binflux,temp_int)*IND2(dydx,1,i)*IND2(dydx,3,i)+\
        IND(binflux,t_int_one)*IND2(dydx,1,i)*IND2(dydx,2,i)+        \
        IND(binflux,t_int_x)*IND2(dydx,0,i)*IND2(dydx,3,i)+        \
        IND(binflux,t_int_x_one)*IND2(dydx,0,i)*IND2(dydx,2,i);
    }

  //  y,x,flux,wbfipmask,binfluxmask,kernel,tup1,binloc,dydx,tup2,issmoothing

  if(PyObject_IsTrue(retbinflux) == 0 && PyObject_IsTrue(retbinstd) == 0)
    {

      //Py_XDECREF(retbinflux);
      //Py_XDECREF(retbinstd);
      Py_XDECREF(binflux);
      Py_XDECREF(binstd);
      return PyArray_Return(output);
    }
  else if (PyObject_IsTrue(retbinflux) == 1 && PyObject_IsTrue(retbinstd)==1)
    {

      //Py_XDECREF(retbinflux);
      //Py_XDECREF(retbinstd);
      return Py_BuildValue("NNN",output,binflux,binstd);
    }
  else if (PyObject_IsTrue(retbinflux) == 1)
    {

      //Py_XDECREF(retbinflux);
      //Py_XDECREF(retbinstd);
      Py_XDECREF(binstd);
      return Py_BuildValue("NN",output,binflux);
    }
  else
    {

      //Py_XDECREF(retbinflux);
      //Py_XDECREF(retbinstd);
      Py_XDECREF(binflux);
      return Py_BuildValue("NN",output,binstd);
    }
}

static char bilinint_doc[]="\
  This function fits the intra-pixel sensitivity effect using bilinear interpolation to fit mean binned flux vs position.  \n\
\n\
    Parameters\n\
    ----------\n\
        ipparams :  tuple\n\
                unused\n\
    y :         1D array, size = # of measurements\n\
                Pixel position along y\n\
    x :         1D array, size = # of measurements\n\
                Pixel position along x\n\
    flux :      1D array, size = # of measurements\n\
                Observed flux at each position\n\
    wherebinflux :  1D array, size = # of bins\n\
                    Measurement number assigned to each bin\n\
    gridpt :    1D array, size = # of measurements\n\
                Bin number in which each measurement is assigned\n\
    dy1 :       1D array, size = # of measurements\n\
                (y - y1)/(y2 - y1)\n\
    dy2 :       1D array, size = # of measurements\n\
                (y2 - y)/(y2 - y1)\n\
    dx1 :       1D array, size = # of measurements\n\
                (x - x1)/(x2 - x1)\n\
    dx2 :       1D array, size = # of measurements\n\
                (x2 - x)/(x2 - x1)\n\
    ysize :     int\n\
                Number of bins along y direction\n\
    xsize :     int\n\
                Number of bins along x direction\n\
    smoothing:  boolean\n\
                Turns smoothing on/off\n\
    \n\
    Returns\n\
    -------\n\
    output :    1D array, size = # of measurements\n\
                Normalized intrapixel-corrected flux multiplier\n\
\n\
    Optional\n\
    --------\n\
    binflux :   1D array, size = # of bins\n\
                Binned Flux values\n\
\n\
    Notes\n\
    -----\n\
    When there are insufficient points for bilinear interpolation, nearest-neighbor interpolation is used.  The code that handles this is in p6models.py.\n\
\n\
    Examples\n\
    --------\n\
    None\n\
\n\
    Revisions\n\
    ---------\n\
    2010-06-11  Kevin Stevenson, UCF\n\
                            kevin218@knights.ucf.edu\n\
                Original version\n\
    2010-07-07  Kevin\n\
                Added wbfipmask\n\
    2011-01-07  nate lust, ucf\n\
                natelust at linux dot com\n\
                Convert to c extension function\n\
";

static PyMethodDef bilinint_methods[] = {
  {"bilinint",(PyCFunction)bilinint,METH_VARARGS|METH_KEYWORDS,bilinint_doc},{NULL}};

static char bilinint_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef bilinint_module=
{
    PyModuleDef_HEAD_INIT,
    "bilinint",             // name of module 
    bilinint_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    bilinint_methods
};

PyMODINIT_FUNC PyInit_bilinint(void)
{
  import_array();
  return PyModule_Create(&bilinint_module);
}

/*
void initbilinint(void)
{
  Py_InitModule("bilinint",bilinint_methods);
  import_array();

}
*/
