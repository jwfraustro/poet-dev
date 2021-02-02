#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *asymcos(PyObject *self, PyObject *args, PyObject *keywds);


static PyObject *asymcos(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *params;
  double c1,c2,c3,c4,c5,P;
  int i;
  npy_intp dims[1];

  static char *kwlist[] = {"params","x","etc",NULL};

  //  etc = PyList_New(0);

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&params,&x,&etc))
    {
      return NULL;
    }

  c1    = IND(params,0);
  c2    = IND(params,1);
  c3    = IND(params,2);
  c4    = IND(params,3);
  c5    = IND(params,4);
  P     = IND(params,5);
  
  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = c1*cos(2*M_PI*(IND(x,i)-c2)/P)+c3*cos(4*M_PI*(IND(x,i)-c4)/P)+c5;
    }


  return PyArray_Return(y);
}

static char asymcos_doc[] = "\
  This function creates an asymmetric cosine model for phase curves.\n\
  See Stevenson et al. (2017), for example.\n\
\n\
  Parameters\n\
  ----------\n\
    c1: amplitude of cosine 1\n\
    c2: offset of cosine 1\n\
    c3: amplitude of cosine 2\n\
    c4: offset of cosine 2\n\
    c5: constant vertical offset\n\
    P:  period (and twice the period) for the cosines\n\
\n\
  Returns\n\
  -------\n\
    This function returns flux values \n\
\n\
  Revisions\n\
  ---------\n\
  2018-06-12: Ryan Challener, UCF\n\
              rchallen@knights.ucf.edu\n\
";


static PyMethodDef asymcos_methods[] = {
  {"asymcos",(PyCFunction)asymcos,METH_VARARGS|METH_KEYWORDS,asymcos_doc},{NULL}};

static char asymcos_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef asymcos_module=
{
    PyModuleDef_HEAD_INIT,
    "asymcos",             // name of module 
    asymcos_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    asymcos_methods
};

PyMODINIT_FUNC PyInit_asymcos(void)
{
  import_array();
  return PyModule_Create(&asymcos_module);
}

/*
void initasymcos(void)
{
  Py_InitModule("asymcos",asymcos_methods);
  import_array();
}
*/
