#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))
#define IND2(a,i,j) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]+j*PyArray_STRIDES(a)[1]))

static PyObject *quadsigip(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *quadsigip(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *out, *ipparams, *sigpos;
  double a,b,c,d,e,f;
  int i;
  npy_intp dims[1];

  static char *kwlist[] = {"ipparams","sigpos","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&ipparams,&sigpos,&etc))
    {
      return NULL;
    }

  a = IND(ipparams,0);
  b = IND(ipparams,1);
  c = IND(ipparams,2);
  d = IND(ipparams,3);
  e = IND(ipparams,4);
  f = IND(ipparams,5);

  dims[0] = PyArray_DIM(sigpos, 1);

  out = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  
  for(i=0;i<dims[0];i++)
    {
      IND(out,i) = a*pow(IND2(sigpos,0,i),2)+b*pow(IND2(sigpos,1,i),2)+  \
                   c*IND2(sigpos,0,i)*IND2(sigpos,1,i)+d*IND2(sigpos,0,i)+e*IND2(sigpos,1,i)+f;
    }
  return PyArray_Return(out);
}

static char quadsigip_doc[]="\
  This function fits the intra-pixel sensitivity effect using a 2D quadratic.\n\
\n\
  Parameters\n\
  ----------\n\
    a: quadratic coefficient in y\n\
    b: quadratic coefficient in x\n\
    c: coefficient for cross-term\n\
    d: linear coefficient in y\n\
    e: linear coefficient in x\n\
    f: constant\n\
\n\
  Returns\n\
  -------\n\
    returns the flux values for the intra-pixel model\n\
\n\
  Revisions\n\
  ---------\n\
  2018-08-17  Ryan Challener   Initial implementation based on\n\
                               quadip.c function.\n\
\n\
";

static PyMethodDef quadsigip_methods[] = {
  {"quadsigip",(PyCFunction)quadsigip,METH_VARARGS|METH_KEYWORDS,quadsigip_doc},{NULL}};

static char quadsigip_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef quadsigip_module=
{
    PyModuleDef_HEAD_INIT,
    "quadsigip",             // name of module 
    quadsigip_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    quadsigip_methods
};

PyMODINIT_FUNC PyInit_quadsigip(void)
{
  import_array();
  return PyModule_Create(&quadsigip_module);
}

/*
void initquadsigip(void)
{
  Py_InitModule("quadsigip",quadsigip_methods);
  import_array();
}
*/
