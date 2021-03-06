#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *fallingexp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *fallingexp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y,*rampparams;
  double goal,m,x0;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O"		\
				  ,kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  goal = IND(rampparams,0);
  m    = IND(rampparams,1);
  x0   = IND(rampparams,2);

  /*
  goal = PyFloat_AsDouble(PyList_GetItem(rampparams,0));
  m    = PyFloat_AsDouble(PyList_GetItem(rampparams,1));
  x0   = PyFloat_AsDouble(PyList_GetItem(rampparams,2));
  */
  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal*(1+exp(-1*m*(IND(x,i)-x0)));
    }

  return PyArray_Return(y);
}

static char fallingexp_doc[] ="\
  This function creates a model that fits a ramp using a falling exponential.\n\
\n\
  Parameters\n\
  ----------\n\
    goal:  goal as x -> inf\n\
    m:	   rise exp\n\
    x0:	   time offset\n\
    x:	   Array of time/phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns an array of y values by combining an eclipse and a rising exponential\n\
\n\
  Revisions\n\
  ---------\n\
  2008-06-16	Kevin Stevenson, UCF  \n\
			kevin218@knights.ucf.edu\n\
		Original version\n\
  2010-12-24    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Updated to C\n\
";

static PyMethodDef fallingexp_methods[] = {
  {"fallingexp",(PyCFunction)fallingexp,METH_VARARGS|METH_KEYWORDS,fallingexp_doc},{NULL}};

static char fallingexp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef fallingexp_module=
{
    PyModuleDef_HEAD_INIT,
    "fallingexp",             // name of module 
    fallingexp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    fallingexp_methods
};

PyMODINIT_FUNC PyInit_fallingexp(void)
{
  import_array();
  return PyModule_Create(&fallingexp_module);
}

/*
void initfallingexp(void)
{
  Py_InitModule("fallingexp",fallingexp_methods);
  import_array();
}
*/
