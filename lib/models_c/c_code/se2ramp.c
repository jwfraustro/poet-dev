#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *se2ramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *se2ramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y,*rampparams;
  double goal,r0,r1,r4,r5,pm0,pm1;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }
  
  goal  = IND(rampparams,0);
  r0    = IND(rampparams,1);
  r1    = IND(rampparams,2);
  pm0   = IND(rampparams,3);
  r4    = IND(rampparams,4);
  r5    = IND(rampparams,5);
  pm1   = IND(rampparams,6);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal + pm0*exp(-r0*IND(x,i) + r1) + pm1*exp(-r4*IND(x,i) + r5);
    }
  return PyArray_Return(y);
}

static char se2ramp_doc[]="\
  This function creates a model that fits a ramp using a rising exponential.\n\
\n\
  Parameters\n\
  ----------\n\
    goal:  goal as x -> inf\n\
    m1,m2: rise exp\n\
    t1,t2: time offset\n\
    t:	   Array of time/phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns an array of y values by combining an eclipse and a rising exponential\n\
\n\
  Revisions\n\
  ---------\n\
  2010-07-30    Kevin Stevenson, UCF  \n\
                kevin218@knights.ucf.edu\n\
                Original version\n\
  2010-12-24    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Converted to C\n\
";

static PyMethodDef se2ramp_methods[] = {
  {"se2ramp",(PyCFunction)se2ramp,METH_VARARGS|METH_KEYWORDS,se2ramp_doc},{NULL}};

static char se2ramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef se2ramp_module=
{
    PyModuleDef_HEAD_INIT,
    "se2ramp",             // name of module 
    se2ramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    se2ramp_methods
};

PyMODINIT_FUNC PyInit_se2ramp(void)
{
  import_array();
  return PyModule_Create(&se2ramp_module);
}

/*
void initse2ramp(void)
{
  Py_InitModule("se2ramp",se2ramp_methods);
  import_array();
}
*/
