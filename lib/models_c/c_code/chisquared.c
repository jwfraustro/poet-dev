#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *chisquared(PyObject *self, PyObject *args);

static PyObject *chisquared(PyObject *self, PyObject *args)
{
  PyArrayObject *mod, *data, *errors;
  int i,lim;
  double chi;

  if(!PyArg_ParseTuple(args,"OOO", &mod, &data, &errors))
    {
      return NULL;
    }

  lim = PyArray_DIMS(mod)[0];
  chi = 0;

  for(i=0;i<lim;i++)
    {
      chi += pow(((IND(mod,i)-IND(data,i))/IND(errors,i)),2);
    }
  return Py_BuildValue("d",chi);
}

static char chisquared_doc[]="\
   This function creates the chi squared statistic given input parameters.\n\
\n\
   Parameters\n\
   ----------\n\
   mod:   1D NPY ARRAY - contains the model to be tested\n\
   data:  1D NPY ARRAY - contains the actual measurements\n\
   errors 1D NPY ARRAY - errors made on the meaurements (not weights)\n\
\n\
   Returns\n\
   -------\n\
   Float - the chi squared value given the model and weights\n\
\n\
   Revisions\n\
   ---------\n\
   2011-01-08    Nate Lust, UCF\n\
                 natelust at linux dot com\n\
                 Initial version, as c extension\n\
";
static char chisquared_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static PyMethodDef chisquared_methods[] = {
  {"chisquared", chisquared, METH_VARARGS, chisquared_doc}, {NULL}
};

static struct PyModuleDef chisquared_module=
{
    PyModuleDef_HEAD_INIT,
    "chisquared",             // name of module 
    chisquared_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    chisquared_methods
};

PyMODINIT_FUNC PyInit_chisquared(void)
{
  import_array();
  return PyModule_Create(&chisquared_module);
}
/*
void initchisquared(void)
{
  Py_InitModule("chisquared", chisquared_methods);
  import_array();
}
*/
