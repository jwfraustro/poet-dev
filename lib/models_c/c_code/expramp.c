#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DIMS(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *expramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *expramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *t,*y, *rampparams;
  double goal,m,a;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","t","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O"		\
				  ,kwlist,&rampparams,&t,&etc))
    {
      return NULL;
    }

  goal = IND(rampparams,0);
  m    = IND(rampparams,1);
  a    = IND(rampparams,2);

  dims[0] = PyArray_DIMS(t)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal-a*exp(-m*IND(t,i));
    }

  return PyArray_Return(y);
}


static PyMethodDef expramp_methods[] = {
  {"expramp",(PyCFunction)expramp,METH_VARARGS|METH_KEYWORDS},{NULL}};

static char expramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef expramp_module=
{
    PyModuleDef_HEAD_INIT,
    "expramp",             // name of module 
    expramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    expramp_methods
};

PyMODINIT_FUNC PyInit_expramp(void)
{
  import_array();
  return PyModule_Create(&expramp_module);
}

/*
void initexpramp(void)
{
  Py_InitModule("expramp",expramp_methods);
  import_array();
}
*/
