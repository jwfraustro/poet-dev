#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *seramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *seramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double goal,r0,r1,pm;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  goal = IND(rampparams,0);
  r0   = IND(rampparams,1);
  r1   = IND(rampparams,2);
  pm   = IND(rampparams,3);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal+pm*exp(-r0*IND(x,i) + r1);
    }
  return PyArray_Return(y);
}

static char seramp_doc[]="\
  This function creates a model that fits a ramp using a rising exponential.\n\
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
  2008-06-24	Kevin Stevenson, UCF  \n\
			kevin218@knights.ucf.edu\n\
		Original version\n\
  2010-12-24    Nate Lust, UCF \n\
                natelust at linux dot com\n\
";

static PyMethodDef seramp_methods[] = {
  {"seramp",(PyCFunction)seramp,METH_VARARGS|METH_KEYWORDS,seramp_doc},{NULL}};

static char seramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef seramp_module=
{
    PyModuleDef_HEAD_INIT,
    "seramp",             // name of module 
    seramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    seramp_methods
};

PyMODINIT_FUNC PyInit_seramp(void)
{
  import_array();
  return PyModule_Create(&seramp_module);
}

/*
void initseramp(void)
{
  Py_InitModule("seramp",seramp_methods);
  import_array();
}
*/
