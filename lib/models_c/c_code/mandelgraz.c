#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *mandelgraz(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *mandelgraz(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *t, *y, *eclparams, *z, *c, *c_norm, *t_sec;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"eclparams","t","etc",NULL};
  
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O"\
				  ,kwlist,&eclparams,&t,&etc))
    {
      return NULL;
    }
 
  double midpt,b,rs,rp,period,maxd,a,flux;
  double days2sec,p_sec,midpt_sec;
  double circum;
  double v;
  double k0,k1,p;

  midpt    = IND(eclparams,0);
  b        = IND(eclparams,1);
  rs       = IND(eclparams,2);
  rp       = IND(eclparams,3);
  period   = IND(eclparams,4);
  maxd     = IND(eclparams,5);
  a        = IND(eclparams,6);
  flux     = IND(eclparams,7);

  //Skip calculation if maxdepth = 0
  if(maxd == 0)
    {
      dims[0] = PyArray_DIMS(t)[0];
      y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
      int i;
      for(i=0;i<dims[0];i++)
	{
	  IND(y,i) = flux;
	}
      return PyArray_Return(y);
    }

  days2sec = 86400;

  p_sec = period * days2sec;

  circum = 2 * M_PI * a; //Assuming circular orbit

  v = circum / p_sec;    //Speed of planet (assuming circular orbit)

  dims[0] = PyArray_DIMS(t)[0];

  midpt_sec = midpt * p_sec;

  t_sec  = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  y      = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  c      = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  c_norm = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  z      = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  int i;
  for(i=0;i<dims[0];i++)
    {
      IND(t_sec, i) =  IND(t,    i) * p_sec;
      IND(c,     i) = (IND(t_sec,i) - midpt_sec) * v;
      IND(c_norm,i) =  IND(c,    i) / rs;
      IND(z,     i) = sqrt(b*b + IND(c_norm,i)*IND(c_norm,i));
    }

  p = rp/rs;
  
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = 0;
      if(IND(z,i) < (1 + p) && IND(z,i) > (1 - p))
	{
	  k0 = acos((  p*p+IND(z,i)*IND(z,i)-1)/2/p/IND(z,i));
	  k1 = acos((1-p*p+IND(z,i)*IND(z,i)  )/2  /IND(z,i));
	  IND(y,i) = 1/M_PI*(k0*p*p+k1
			     -sqrt((4*IND(z,i)*IND(z,i)-pow((1+IND(z,i)*IND(z,i)-p*p),2))/4));
	}
      else if(IND(z,i)<=(1-p))
	{
	  IND(y,i) = p*p;
	}

      IND(y,i) = flux*(1-maxd*rs*rs/rp/rp*IND(y,i));
      
    }
  Py_XDECREF(dims);
  Py_XDECREF(t_sec);
  Py_XDECREF(c);
  Py_XDECREF(z);
  Py_XDECREF(c_norm);
  return PyArray_Return(y);
}

static char mandelgraz_doc[] ="\
  This function computes the secondary eclipse shape using equations provided by Mandel & Agol (2002) with modifications for a grazing eclipse.\n\
\n\
  Parameters\n\
  ----------\n\
    midpt:    Center of eclipse\n\
    b:        Impact parameter\n\
    rs:       Radius of host star in meters\n\
    period:   Planet orbital period in days\n\
    maxdepth: Eclipse depth if the orbit was not grazing\n\
    a:        Semimajor axis in meters\n\
    flux:     System flux\n\
    t:	      Array of phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns the flux for each point in t.\n\
\n\
  Revisions\n\
  ---------\n\
  2018-05-03    Ryan Challener, UCF \n\
                rchallen@knights.ucf.edu\n\
                Original version, adapted from\n\
                mandelecl.c\n\
\n\
";

static PyMethodDef mandelgraz_methods[]={
  {"mandelgraz",(PyCFunction)mandelgraz,METH_VARARGS|METH_KEYWORDS,mandelgraz_doc}, \
  {NULL}};

static char mandelgraz_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef mandelgraz_module=
{
    PyModuleDef_HEAD_INIT,
    "mandelgraz",             // name of module 
    mandelgraz_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    mandelgraz_methods
};

PyMODINIT_FUNC PyInit_mandelgraz(void)
{
  import_array();
  return PyModule_Create(&mandelgraz_module);
}

/*
void initmandelgraz(void)
{
  Py_InitModule("mandelgraz",mandelgraz_methods);
  import_array();
}
*/
