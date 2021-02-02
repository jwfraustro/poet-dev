#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *re2ramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *re2ramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *t,*y,*rampparams;
  double goal,a,b,m1,m2,t1,t2;
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
  a    = IND(rampparams,1);
  m1   = IND(rampparams,2);
  t1   = IND(rampparams,3);
  b    = IND(rampparams,4);
  m2   = IND(rampparams,5);
  t2   = IND(rampparams,6);

  dims[0] = PyArray_DIMS(t)[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal-a*exp(-m1*(IND(t,i)-t1))-b*exp(-m2*(IND(t,i)-t2));
    }
  return PyArray_Return(y);
}

static char re2ramp_doc[]="\
  This function creates a model that fits a ramp using a rising exponential.\n\
\n\
  Parameters\n\
  ----------\n\
    goal:  goal as x -> inf\n\
    m1,m2: rise exp\n\
    t1,t2: time offset\n\
    t:	   Array of time/phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns an array of y values by combining an eclipse and a rising exponential\n\
\n\
  Revisions\n\
  ---------\n\
  2010-07-30    Kevin Stevenson, UCF  \n\
                kevin218@knights.ucf.edu\n\
                Original version\n\
  2010-12-24    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Converted to C\n\
";

static PyMethodDef re2ramp_methods[] = {
  {"re2ramp",(PyCFunction)re2ramp,METH_VARARGS|METH_KEYWORDS,re2ramp_doc},{NULL}};

static char re2ramp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef re2ramp_module=
{
    PyModuleDef_HEAD_INIT,
    "re2ramp",             // name of module 
    re2ramp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    re2ramp_methods
};

PyMODINIT_FUNC PyInit_re2ramp(void)
{
  import_array();
  return PyModule_Create(&re2ramp_module);
}

/*
void initre2ramp(void)
{
  Py_InitModule("re2ramp",re2ramp_methods);
  import_array();
}
*/
