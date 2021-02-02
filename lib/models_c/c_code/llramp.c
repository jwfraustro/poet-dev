#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *llramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *llramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double x0,a,b,c,x1;
  int i;
  npy_intp dims[1];

  //etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  x0 = IND(rampparams,0);
  a  = IND(rampparams,1);
  b  = IND(rampparams,2);
  c  = IND(rampparams,3);
  x1 = IND(rampparams,4);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = 1;
      if(IND(x,i)>x0)
	{
	  IND(y,i) = a*log(IND(x,i)-x0)+b*(IND(x,i)-x1)+c;
	}
    }

  return PyArray_Return(y);
}

static char llramp_doc[]="\
  This function creates a model that fits a ramp using a log + linear ploynomial.\n\
\n\
  Parameters\n\
  ----------\n\
    x0: phase offset for log term\n\
    a:  log(x) constant\n\
    b:  x constant\n\
    c:  x=0 offset\n\
    x1: phase offset for polynomial\n\
    x:  Array of time/phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns the flux values for the ramp models\n\
\n\
  Revisions\n\
  ---------\n\
  2008-08-31	Kevin Stevenson, UCF  	\n\
                kevin218@knights.ucf.edu\n\
                Original version\n\
  2010-07-07    Kevin Stevenson\n\
                New code for when x < x0\n\
  2010-12-26    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Updated to C extension\n\
";

static PyMethodDef llramp_methods[] = {
  {"llramp",(PyCFunction)llramp,METH_VARARGS|METH_KEYWORDS,llramp_doc},{NULL}};

static char llramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef llramp_module=
{
    PyModuleDef_HEAD_INIT,
    "llramp",             // name of module 
    llramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    llramp_methods
};

PyMODINIT_FUNC PyInit_llramp(void)
{
  import_array();
  return PyModule_Create(&llramp_module);
}

/*
void initllramp(void)
{
  Py_InitModule("llramp",llramp_methods);
  import_array();
}
*/
