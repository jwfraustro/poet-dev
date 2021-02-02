#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))

static PyObject *trnlldsp(PyObject *self, PyObject *args, PyObject *keywd);

static PyObject *trnlldsp(PyObject *self, PyObject *args, PyObject *keywd)
{
  PyObject *etc;
  PyArrayObject *t, *y, *params;
  npy_intp dims[1];
  
  static char *kwlist[] = {"params","t","etc",NULL};
  
  if(!PyArg_ParseTupleAndKeywords(args,keywd,"OO|O",kwlist,	\
				  &params,&t,&etc))
    {
      return NULL;
    }
 
  double midpt,rprs,cosi,ars,flux,x;
  double p,z,c1,c2,c3,c4,Sigma4,I1star,sig1,sig2,I2star,mod;

  midpt = IND(params,0);
  rprs  = IND(params,1);
  cosi  = IND(params,2);
  ars   = IND(params,3);
  flux  = IND(params,4);
  p     = IND(params,5);
  c1    = IND(params,6);
  c2    = IND(params,7);
  c3    = IND(params,8);
  c4    = IND(params,9);

  dims[0] = PyArray_DIMS(t)[0];
 
  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  Sigma4 = (1-c1/5-c2/3-3*c3/7-c4/2);

  int i;
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = 1;
      if(rprs!=0)
        {
          mod = (IND(t,i)-midpt)-floor((IND(t,i)-midpt)/p)*p;
          if((mod > (p/4)) && (mod < (3*p/4)))
            {
              z = ars;
            }
          else
            {
              z = ars*sqrt(pow(sin(2*M_PI*(IND(t,i)-midpt)/p),2)+pow(cosi*cos(2*M_PI \
                  *(IND(t,i)-midpt)/p),2));
            }
          //Ingress or egress
          if(z>(1-rprs) && z<=(1+rprs))
            {
              x        = 1 - pow((z-rprs),2);
              I1star   = 1 - c1*(1-4/5.0*sqrt(sqrt(x)))        \
                           - c2*(1-2/3.0*sqrt(x))              \
                           - c3*(1-4/7.0*sqrt(sqrt(x*x*x)))    \
                           - c4*(1-4/8.0*x);
              IND(y,i) = 1 - I1star*(rprs*rprs*acos((z-1)/rprs) \
                           - (z-1)*sqrt(rprs*rprs-(z-1)*(z-1)))/M_PI/Sigma4;
            }
          //t2 - t3 (except @ z=0)
          else if(z<=(1-rprs) && z!=0)
            {
              sig1    = sqrt(sqrt(1-pow((z-rprs),2)));
              sig2    = sqrt(sqrt(1-pow((z+rprs),2)));
              I2star  = 1 - c1*(1+(pow(sig2,5)-pow(sig1,5))/5.0/rprs/z)   \
                          - c2*(1+(pow(sig2,6)-pow(sig1,6))/6.0/rprs/z)   \
                          - c3*(1+(pow(sig2,7)-pow(sig1,7))/7.0/rprs/z)   \
                          - c4*(rprs*rprs+z*z);
              IND(y,i) = 1-rprs*rprs*I2star/Sigma4;
            }
          //z=0 (midpoint)
          else if(z==0)
            {
              IND(y,i)=1-rprs*rprs/Sigma4;
            }
        }
      IND(y,i) *= flux;
    }
  return PyArray_Return(y);
}

static char trnlldsp_doc[] ="\
  This function computes the primary transit shape using non-linear limb-darkening equations for a 'small planet' (rprs <= 0.1), as provided by Mandel & Agol (2002).\n\
\n\
  Parameters\n\
  ----------\n\
    midpt:  Center of eclipse\n\
    rprs:   Planet radius / stellar radius\n\
    cosi:   Cosine of the inclination\n\
    ars:    Semi-major axis / stellar radius\n\
    flux:   Flux offset from 0\n\
    c#:     Limb-darkening coefficients\n\
    t:            Array of phase/time points\n\
    p:      Period in same units as t\n\
\n\
  Returns\n\
  -------\n\
    This function returns the flux for each point in t.\n\
\n\
  References\n\
  ----------\n\
\n\
  Mandel & Agol (2002)\n\
  /home/esp01/doc/Mandel+Agol-2002_eq8.pdf\n\
  /home/esp01/code/MandelAgol/occultsmall.pro\n\
\n\
  Revisions\n\
  ---------\n\
  2010-12-15        Kevin Stevenson, UCF  \n\
                kevin218@knights.ucf.edu\n\
                Converted to Python\n\
  2010-12-24    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Converted to C\n\
";

static PyMethodDef trnlldsp_methods[]={
  {"trnlldsp",(PyCFunction)trnlldsp,METH_VARARGS|METH_KEYWORDS,trnlldsp_doc},	\
  {NULL}};

static char trnlldsp_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef trnlldsp_module=
{
    PyModuleDef_HEAD_INIT,
    "trnlldsp",             // name of module 
    trnlldsp_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    trnlldsp_methods
};

PyMODINIT_FUNC PyInit_trnlldsp(void)
{
  import_array();
  return PyModule_Create(&trnlldsp_module);
}

/*
void inittrnlldsp(void)
{
  Py_InitModule("trnlldsp",trnlldsp_methods);
  import_array();
}
*/
