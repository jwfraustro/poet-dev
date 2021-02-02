#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *quadramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *quadramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double a,b,c,x0;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  a  = IND(rampparams,0);
  b  = IND(rampparams,1);
  c  = IND(rampparams,2);
  x0 = IND(rampparams,3);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = a*pow((IND(x,i)-x0),2)+b*(IND(x,i)-x0)+c;
    }
  return PyArray_Return(y);
}

static char quadramp_doc[] = "\
 NAME:\n\
	QUADRAMP\n\
\n\
 PURPOSE:\n\
	This function creates a model that fits a quadratically ramped eclipse\n\
\n\
 CATEGORY:\n\
	Astronomy.\n\
\n\
 CALLING SEQUENCE:\n\
\n\
	Result = QUADRAMP([midpt,width,depth,a,b,c],x)\n\
\n\
 INPUTS:\n\
    	midpt:	Midpoint of eclipse\n\
	width:	Eclipse durations\n\
	depth:	Depth of eclipse\n\
	a:	x^2 constant\n\
	b:	x constant\n\
	c:	x=0 offset\n\
	x0: time/phase offset (constant)\n\
	x:	Array of time/phase points\n\
\n\
 OUTPUTS:\n\
	This function returns an array of y values by combining an eclipse and a quadratic\n\
\n\
 PROCEDURE:\n\
\n\
 EXAMPLE:\n\
\n\
\n\
\n\
 MODIFICATION HISTORY:\n\
 	Written by:	Kevin Stevenson, UCF  	2008-06-22\n\
			kevin218@knights.ucf.edu\n\
                        Nate Lust, UCF          2010-12-25\n\
                        natelust at linux dot com\n\
                        converted to c\n\
\n\
";

static PyMethodDef quadramp_methods[] = {
  {"quadramp",(PyCFunction)quadramp,METH_VARARGS|METH_KEYWORDS,quadramp_doc},{NULL}};

static char quadramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef quadramp_module=
{
    PyModuleDef_HEAD_INIT,
    "quadramp",             // name of module 
    quadramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    quadramp_methods
};

PyMODINIT_FUNC PyInit_quadramp(void)
{
  import_array();
  return PyModule_Create(&quadramp_module);
}

/*
void initquadramp(void)
{
  Py_InitModule("quadramp",quadramp_methods);
  import_array();
}
*/
