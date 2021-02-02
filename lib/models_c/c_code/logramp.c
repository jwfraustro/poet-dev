#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *logramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *logramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double x0,a,b,c,d,e;
  int i;
  npy_intp dims[1];

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  //etc = PyList_New(0);

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  x0 = IND(rampparams,0);
  a  = IND(rampparams,1);
  b  = IND(rampparams,2);
  c  = IND(rampparams,3);
  d  = IND(rampparams,4);
  e  = IND(rampparams,5);

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      if(IND(x,i)<=x0)
        IND(y,i) = e;
      else
	    IND(y,i) = a*pow(log(IND(x,i)-x0),4)+b*pow(log(IND(x,i)-x0),3) \
        +c*pow(log(IND(x,i)-x0),2)+d*log(IND(x,i)-x0)+e;
    }
  return PyArray_Return(y);
}

static char logramp_doc[]="\
 NAME:\n\
	LOGRAMP\n\
\n\
 PURPOSE:\n\
	This function creates a model that fits a natural log + linear ramped eclipse\n\
\n\
 CATEGORY:\n\
	Astronomy.\n\
\n\
 CALLING SEQUENCE:\n\
\n\
	Result = LOGRAMP([midpt,width,depth,x12,x34,x0,b,c],x)\n\
\n\
 INPUTS:\n\
    	midpt:	Midpoint of eclipse\n\
	width:	Eclipse durations\n\
	depth:	Depth of eclipse\n\
	x12:	Ingress time\n\
	x34:	Egress time\n\
	x0:	time offset\n\
	b:	x constant\n\
	c:	x=0 offset\n\
	x:	Array of time/phase points\n\
\n\
 OUTPUTS:\n\
	This function returns an array of y values by combining an eclipse and the ramp model\n\
\n\
 PROCEDURE:\n\
\n\
 EXAMPLE:\n\
\n\
\n\
\n\
 MODIFICATION HISTORY:\n\
 	Written by:	Kevin Stevenson, UCF  	2008-06-26\n\
			kevin218@knights.ucf.edu\n\
";

static PyMethodDef logramp_methods[] = {
  {"logramp",(PyCFunction)logramp,METH_VARARGS|METH_KEYWORDS,logramp_doc},{NULL}};

static char logramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef logramp_module=
{
    PyModuleDef_HEAD_INIT,
    "logramp",             // name of module 
    logramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    logramp_methods
};

PyMODINIT_FUNC PyInit_logramp(void)
{
  import_array();
  return PyModule_Create(&logramp_module);
}

/*
void initlogramp(void)
{
  Py_InitModule("logramp",logramp_methods);
  import_array();
}
*/
