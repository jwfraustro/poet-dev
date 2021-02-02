#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *relramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *relramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double goal,m,x0,a,b,x1;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  goal = IND(rampparams,0);
  m    = IND(rampparams,1);
  x0   = IND(rampparams,2);
  a    = IND(rampparams,3);
  b    = IND(rampparams,4);
  x1   = IND(rampparams,5);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal*(1-exp(-1*m*(IND(x,i)-x0)))+a*(IND(x,i)-x1)+b;
    }
  return PyArray_Return(y);
}

static PyMethodDef relramp_methods[] = {
  {"relramp",(PyCFunction)relramp,METH_VARARGS|METH_KEYWORDS},{NULL}};

static char relramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef relramp_module=
{
    PyModuleDef_HEAD_INIT,
    "relramp",             // name of module 
    relramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    relramp_methods
};

PyMODINIT_FUNC PyInit_relramp(void)
{
  import_array();
  return PyModule_Create(&relramp_module);
}

/*
void initrelramp(void)
{
  Py_InitModule("relramp",relramp_methods);
  import_array();
}
*/
