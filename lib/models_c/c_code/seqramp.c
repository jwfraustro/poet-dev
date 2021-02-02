#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *seqramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *seqramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double goal,r0,r1,r2,r3,x0,pm;
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
  r2   = IND(rampparams,3);
  r3   = IND(rampparams,4);
  x0   = IND(rampparams,5);
  pm   = IND(rampparams,6);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal + pm*exp(-r0*IND(x,i)+r1) + r2*(IND(x,i)-x0) + r3*pow((IND(x,i)-x0),2);
    }
  return PyArray_Return(y);
}

static PyMethodDef seqramp_methods[] = {
  {"seqramp",(PyCFunction)seqramp,METH_VARARGS|METH_KEYWORDS},{NULL}};

static char seqramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef seqramp_module=
{
    PyModuleDef_HEAD_INIT,
    "seqramp",             // name of module 
    seqramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    seqramp_methods
};

PyMODINIT_FUNC PyInit_seqramp(void)
{
  import_array();
  return PyModule_Create(&seqramp_module);
}

/*
void initseqramp(void)
{
  Py_InitModule("seqramp",seqramp_methods);
  import_array();
}
*/
