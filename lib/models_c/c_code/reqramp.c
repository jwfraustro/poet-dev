#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *reqramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *reqramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double goal,m,x0,a,b,c,x1;
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
  c    = IND(rampparams,5);
  x1   = IND(rampparams,6);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal*(1-exp(-1*m*(IND(x,i)-x0)))+a*pow((IND(x,i)-x1),2)	\
						       +b*(IND(x,i)-x1)+c;
    }
  return PyArray_Return(y);
}

static PyMethodDef reqramp_methods[] = {
  {"reqramp",(PyCFunction)reqramp,METH_VARARGS|METH_KEYWORDS},{NULL}};

static char reqramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef reqramp_module=
{
    PyModuleDef_HEAD_INIT,
    "reqramp",             // name of module 
    reqramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    reqramp_methods
};

PyMODINIT_FUNC PyInit_reqramp(void)
{
  import_array();
  return PyModule_Create(&reqramp_module);
}

/*
void initreqramp(void)
{
  Py_InitModule("reqramp",reqramp_methods);
  import_array();
}
*/
