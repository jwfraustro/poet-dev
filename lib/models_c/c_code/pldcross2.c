#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]))
#define IND2(a,i,j) *((double *)(PyArray_DATA(a)+i*PyArray_STRIDES(a)[0]+j*PyArray_STRIDES(a)[1]))

static PyObject *pldcross2(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *pldcross2(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  // We don't actually need t, but it's included here for consistency
  // with other models
  PyArrayObject *phat, *y, *t, *pixparams;
  npy_intp dims[2];

  static char *kwlist[] = {"pixparams","t","phat","etc",NULL};
  
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OOO|O"\
				  ,kwlist,&pixparams,&t,&phat,&etc))
    {
      return NULL;
    }

  int i,j,k,l;
  //i is frame index
  //j and k are pixel indices
  //l keeps track of parameter number
  dims[0] = PyArray_DIMS(phat)[0];
  dims[1] = PyArray_DIMS(phat)[1];

  npy_intp outdims[1];
  outdims[0] = PyArray_DIMS(phat)[0];
  
  y = (PyArrayObject *) PyArray_SimpleNew(1,outdims,NPY_DOUBLE);
  
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = 0;
    }

  l = 0;
  for(j=0;j<dims[1];j++)
    {
      for(k=j;k<dims[1];k++)
	{
	  for(i=0;i<dims[0];i++)
	    {
	      IND(y,i) = IND(y,i) + (IND(pixparams,l) * IND2(phat,i,j) * IND2(phat,i,k));
	    }
	  l += 1;
	}
    }

  Py_XDECREF(dims);
  Py_XDECREF(outdims);
  return PyArray_Return(y);
}

static char pldcross2_doc[] ="\
  This function computes the 2nd-order cross terms of the PLD model\n\
  as described in Deming et al. (2015).\n\
\n\
  Parameters\n\
  ----------\n\
    pixparams: float ndarray\n\
               A weight for each pixel combination\n\
    t:         float ndarray\n\
               Time array. Unused, but included for consistency\n\
               with other model functions.\n\
    phat:      float 2D ndarray\n\
               phat as defined in Deming et al. (2015). Array\n\
               is nframes x npix.\n\
\n\
  Returns\n\
  -------\n\
    This function returns the flux for each frame.\n\
\n\
  Revisions\n\
  ---------\n\
  2018-07-17    Ryan Challener, UCF \n\
                rchallen@knights.ucf.edu\n\
                Original version.\n\
\n\
";

static PyMethodDef pldcross2_methods[]={
  {"pldcross2",(PyCFunction)pldcross2,METH_VARARGS|METH_KEYWORDS,pldcross2_doc}, \
  {NULL}};

static char pldcross2_module_doc[]="\
This is a test, I wonder were this will show up.\
";

static struct PyModuleDef pldcross2_module=
{
    PyModuleDef_HEAD_INIT,
    "pldcross2",             // name of module 
    pldcross2_module_doc, // module documentation, may be NULL 
    -1,                     // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    pldcross2_methods
};

PyMODINIT_FUNC PyInit_pldcross2(void)
{
  import_array();
  return PyModule_Create(&pldcross2_module);
}

/*
void initpldcross2(void)
{
  Py_InitModule("pldcross2",pldcross2_methods);
  import_array();
}
*/
