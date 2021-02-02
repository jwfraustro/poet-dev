#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *deramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *deramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double g,r0,r1,th0,th1,pm,goal,a,b,gb,r0b,r1b;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  g    = IND(rampparams,0);
  r0   = IND(rampparams,1);
  r1   = IND(rampparams,2);
  th0  = IND(rampparams,3);     //Angle b/w r0 & r1
  th1  = IND(rampparams,4);     //Angle b/w r0 & g
  pm   = IND(rampparams,5);
  gb   = IND(rampparams,6);     //Best-fit value
  r0b  = IND(rampparams,7);     //Best-fit value
  r1b  = IND(rampparams,8);     //Best-fit value
  
  a    =  r0*cos(th1)*cos(th0) - r1*cos(th1)+sin(th0) + g*sin(th1) + r0b;
  b    =  r0*sin(th0)          + r1*cos(th0)                       + r1b;
  goal = -r0*sin(th1)*cos(th0) + r1*sin(th1)*sin(th0) + g*cos(th1) + gb;

  dims[0] = PyArray_DIMS(x)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      // IND(y,i) = goal+pm*exp(-(IND(x,i)*cost-sint)/r0 + (IND(x,i)*sint+cost)*r1);
      IND(y,i) = goal+pm*exp(-a*IND(x,i) + b);
    }
  return PyArray_Return(y);
}

static char deramp_doc[]="\
  This function creates a model that fits a ramp using a rising exponential.\n\
\n\
  Parameters\n\
  ----------\n\
    goal:  goal as x -> inf\n\
    m:	   rise exp\n\
    x0:	   time offset\n\
    x:	   Array of time/phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns an array of y values by combining an eclipse and a rising exponential\n\
\n\
  Revisions\n\
  ---------\n\
  2008-06-24	Kevin Stevenson, UCF  \n\
			kevin218@knights.ucf.edu\n\
		Original version\n\
  2010-12-24    Nate Lust, UCF \n\
                natelust at linux dot com\n\
";

static PyMethodDef deramp_methods[] = {
  {"deramp",(PyCFunction)deramp,METH_VARARGS|METH_KEYWORDS,deramp_doc},{NULL}};

static char deramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef deramp_module=
{
    PyModuleDef_HEAD_INIT,
    "deramp",             // name of module 
    deramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    deramp_methods
};

PyMODINIT_FUNC PyInit_deramp(void)
{
  import_array();
  return PyModule_Create(&deramp_module);
}

/*
void initderamp(void)
{
  Py_InitModule("deramp",deramp_methods);
  import_array();
}
*/
