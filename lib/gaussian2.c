#include "Python.h"
#include <numpy/arrayobject.h>
#include <math.h>
#include <sys/types.h>

// The i,j value of the array a:
#define IND(a,i,j) *((double *)(a->data + i*a->strides[0] + j*a->strides[1]))


static PyObject *make_gauss(PyObject *self, PyObject *args) {
    double x0, y0, sigmax, sigmay, background;
    double height=0.0;
    PyArrayObject *mat;
    int dims[2];
    int i, j, n, m;

    if (!PyArg_ParseTuple(args, "iidddd|dd", &n, &m, &x0, &y0, &sigmax, 
                          &sigmay, &height, &background))
        return NULL;

    if (height == 0.0)
        height = 1.0 / (sqrt(2.0 * 3.14159) * sigmax * sigmay);

    dims[0] = n;
    dims[1] = m;
    mat = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
 
    for(  i=0; i<n; i++){
        for(j=0; j<m; j++){
            IND(mat,i,j) = height * exp(-0.5*( pow( (i-x0)/sigmax, 2.0) +
                                           pow( (j-y0)/sigmay, 2.0))) +
                                           background; 
        }
    }

    return mat;
}

// The module doc string
PyDoc_STRVAR(gaussian2_module__doc__,
  "Gaussian point evalutation kernel");

// The function doc string
PyDoc_STRVAR(make_gauss__doc__,
  "x,y,max_iterations -> iteration count at that point, up to max_iterations");

// A list of all the methods defined by this module.
// "make_gauss" is the name seen inside of Python
// make_gauss is the name of the C function handling the Python call
// "METH_VARGS" tells Python how to call the handler
// The {NULL, NULL...} entry indicates the end of the method definitions
static PyMethodDef gaussian2_methods[] = {
    {"make_gauss", make_gauss, METH_VARARGS, make_gauss__doc__},
    {NULL, NULL, 0, NULL}      /* sentinel */
};

static struct PyModuleDef gaussian2_module =
{
    PyModuleDef_HEAD_INIT,
    "gaussian2",             // name of module 
    gaussian2_module__doc__, // module documentation, may be NULL 
    -1,                      // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    gaussian2_methods
};

// When Python imports a C module named 'X' it loads the module
// then looks for a method named "PyInit_"+X and calls it.  Hence
// for the module "mandelbrot" the initialization function is
// "PyInit_mandelbrot".  The PyMODINIT_FUNC helps with portability
// across operating systems and between C and C++ compilers

PyMODINIT_FUNC PyInit_gaussian2(void)
{
    import_array();
    return PyModule_Create(&gaussian2_module);
}
