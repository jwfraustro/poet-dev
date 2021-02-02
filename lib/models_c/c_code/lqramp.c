#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *lqramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *lqramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double x0,a,b,c,d,x1,temp;
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
  d  = IND(rampparams,4);
  x1 = IND(rampparams,5);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      if(IND(x,i)<=x0)
        IND(y,i) = 1;
      else
        IND(y,i) = a*log(IND(x,i)-x0)+b*pow((IND(x,i)-x1),2)+c*(IND(x,i)-x1)+d;
    }
  return PyArray_Return(y);
}

static char lqramp_doc[]="\
  NAME:\n\
	LQRAMP\n\
\n\
 PURPOSE:\n\
	This function creates a model that fits a log + quadratic ramped eclipse\n\
\n\
 CATEGORY:\n\
	Astronomy.\n\
\n\
 CALLING SEQUENCE:\n\
\n\
	Result = LQRAMP([a,b,c,d],x)\n\
\n\
 INPUTS:\n\
    x0: phase offset\n\
	a:	log(x) term\n\
	b:	quadratic term\n\
	c:	linear term\n\
    d:  constant term\n\
	x:	Array of time/phase points\n\
\n\
 OUTPUTS:\n\
	This function returns an array of y values...\n\
\n\
 EXAMPLE:\n\
\n\
\n\
 MODIFICATION HISTORY:\n\
 	Written by:	Kevin Stevenson, UCF  	2009-11-28\n\
			kevin218@knights.ucf.edu\n\
\n\
";

static PyMethodDef lqramp_methods[] = {
  {"lqramp",(PyCFunction)lqramp,METH_VARARGS|METH_KEYWORDS,lqramp_doc},{NULL}};

static char lqramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef lqramp_module=
{
    PyModuleDef_HEAD_INIT,
    "lqramp",             // name of module 
    lqramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    lqramp_methods
};

PyMODINIT_FUNC PyInit_lqramp(void)
{
  import_array();
  return PyModule_Create(&lqramp_module);
}

/*
void initlqramp(void)
{
  Py_InitModule("lqramp",lqramp_methods);
  import_array();
}
*/
