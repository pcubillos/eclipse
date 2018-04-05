#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>
#include<math.h>

#include "ind.h"


double sign(double x){
  if (x<0)
    return -1.0;
  return 1.0;
}


PyDoc_STRVAR(mandelecl__doc__,
"Secondary-eclipse light-curve model from Mandel & Agol (2002).\n\
                                                               \n\
Parameters                                                     \n\
----------                                                     \n\
params: 1D float ndarray                                       \n\
   Eclipse model parameters:                                   \n\
     midpt:  Center of eclipse.                                \n\
     width:  Eclipse duration between contacts 1 to 4.         \n\
     depth:  Eclipse depth.                                    \n\
     t12:    Eclipse duration between contacts 1 to 2.         \n\
     t34:    Eclipse duration between contacts 3 to 4.         \n\
     flux:   Out-of-eclipse flux level.                        \n\
t: 1D float ndarray                                            \n\
   The lightcurve's phase/time points.                         \n\
                                                               \n\
Returns                                                        \n\
-------                                                        \n\
eclipse: 1D float ndarray                                      \n\
   Mandel & Agol eclipse model evaluated at points t.");

static PyObject *mandelecl(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, depth, t12, t34, flux;
  double t1, t2, t3, t4, p, z, k0, k1;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt = INDd(params, 0);
  width = INDd(params, 1);
  depth = INDd(params, 2);
  t12   = INDd(params, 3);
  t34   = INDd(params, 4);
  flux  = INDd(params, 5);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  if(depth == 0){
      for(i=0; i<dims[0]; i++){
        INDd(eclipse, i) = flux;
      }
      return Py_BuildValue("N", eclipse);
    }

  /* Time of contact points: */
  t1 = midpt - width/2;

  if ((t1+t12) < midpt)
    t2 = t1 + t12;
  else
    t2 = midpt;

  t4 = midpt + width/2;
  if ((t4-t34) > midpt)
    t3 = t4 - t34;
  else
    t3 = midpt;

  p = depth/sqrt(fabs(depth));

  for(i=0; i<dims[0]; i++){
    /* Out of eclipse: */
    if (INDd(t,i) < t1 || INDd(t,i) > t4){
      INDd(eclipse,i) = 1.0;
    }
    /* Totality:       */
    else if (INDd(t,i) >= t2  &&  INDd(t,i) <= t3){
      INDd(eclipse,i) = 1 - depth;
    }
    /* Eq. (1) of Mandel & Agol (2002) for ingress/egress:  */
    else if (p != 0){
      if (INDd(t,i) > t1  &&  INDd(t,i) < t2){
        z  = -2*p*(INDd(t,i)-t1)/t12 + 1 + p;
        k0 = acos(0.5*(p*p + z*z - 1)/p/z);
        k1 = acos(0.5*(1 - p*p + z*z)/z);
        INDd(eclipse,i) = 1 - depth/fabs(depth)/M_PI * (p*p*k0 + k1
                                    - sqrt((4*z*z - pow((1+z*z-p*p),2))/4));
      }
      else if (INDd(t,i) > t3  &&  INDd(t,i) < t4){
        z  = 2*p*(INDd(t,i)-t3)/t34 + 1 - p;
        k0 = acos((p*p+z*z-1)/2/p/z);
        k1 = acos((1-p*p+z*z)/2/z);
        INDd(eclipse,i) = 1-depth/fabs(depth)/M_PI*(p*p*k0 + k1
                                    - sqrt((4*z*z - pow((1+z*z-p*p),2))/4));
      }
    }
    INDd(eclipse,i) *= flux;
  }

  return Py_BuildValue("N", eclipse);
}


PyDoc_STRVAR(eclipse_flat__doc__,
"Secondary-eclipse light-curve model with flat baseline and\n\
independent ingress and egress depths.");

static PyObject *eclipse_flat(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, idepth, edepth, t12, t34, flux;
  double t1, t2, t3, t4, pi, pe, z, k0, k1;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt  = INDd(params, 0);
  width  = INDd(params, 1);
  idepth = INDd(params, 2);  // Ingress depth
  edepth = INDd(params, 3);  // Egress depth
  t12    = INDd(params, 4);
  t34    = INDd(params, 5);
  flux   = INDd(params, 6);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  if(idepth == 0){
      for(i=0; i<dims[0]; i++){
        INDd(eclipse, i) = flux;
      }
      return Py_BuildValue("N", eclipse);
    }

  /* Time of contact points: */
  t1 = midpt - width/2;
  t2 = t1 + t12;
  /* Grazing eclipse:        */
  if ((t1+t12) > midpt)
    t2 = midpt;

  t4 = midpt + width/2;
  t3 = t4 - t34;
  if ((t4-t34) < midpt)
    t3 = midpt;

  /* Rp/Rs at ingress and egress: */
  pi = idepth/sqrt(fabs(idepth));  /* Not to confuse with pi */
  pe = edepth/sqrt(fabs(edepth));

  for(i=0; i<dims[0]; i++){
    /* Before ingress: */
    if (INDd(t,i) < t1){
      INDd(eclipse,i) = 1.0 + idepth;
    }
    /* During ingress:                   */
    /* Eq. (1) of Mandel & Agol (2002):  */
    else if (INDd(t,i) < t2){
      z  = -2*pi*(INDd(t,i)-t1)/t12 + 1 + pi;
      k0 = acos(0.5*(pi*pi + z*z - 1)/pi/z);
      k1 = acos(0.5*(1 - pi*pi + z*z)/z);
      INDd(eclipse,i) = 1 + idepth - sign(idepth)/M_PI * (pi*pi*k0 + k1
                                   - sqrt((z*z - 0.25*pow((1+z*z-pi*pi),2))));
    }
    /* Totality:       */
    else if (INDd(t,i) < t3){
      INDd(eclipse,i) = 1.0;
    }
    /* During egress:  */
    else if (INDd(t,i) < t4){
      z  = 2*pe*(INDd(t,i)-t3)/t34 + 1 - pe;
      k0 = acos((pe*pe+z*z-1)/2/pe/z);
      k1 = acos((1-pe*pe+z*z)/2/z);
      INDd(eclipse,i) = 1 + edepth - sign(edepth)/M_PI*(pe*pe*k0 + k1
                                   - sqrt((z*z - 0.25*pow((1+z*z-pe*pe),2))));
    }
    /* After egress:   */
    else if (INDd(t,i) >= t4){
      INDd(eclipse,i) = 1.0 + edepth;
    }
    INDd(eclipse,i) *= flux;
  }
  return Py_BuildValue("N", eclipse);
}


PyDoc_STRVAR(eclipse_lin__doc__,
"Secondary-eclipse light-curve model with linear baseline and \n\
independent ingress and egress depths and slopes.");

static PyObject *eclipse_lin(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, idepth, edepth, t12, t34, flux, islope, eslope;
  double t1, t2, t3, t4, pi, pe, z, k0, k1;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt  = INDd(params, 0);
  width  = INDd(params, 1);
  idepth = INDd(params, 2);  // Ingress depth
  edepth = INDd(params, 3);  // Egress depth
  t12    = INDd(params, 4);
  t34    = INDd(params, 5);
  flux   = INDd(params, 6);
  islope = INDd(params, 7);
  eslope = INDd(params, 8);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  if(idepth == 0){
      for(i=0; i<dims[0]; i++){
        INDd(eclipse, i) = flux;
      }
      return Py_BuildValue("N", eclipse);
    }

  /* Time of contact points: */
  t1 = midpt - width/2;
  t2 = t1 + t12;
  /* Grazing eclipse:        */
  if ((t1+t12) > midpt)
    t2 = midpt;

  t4 = midpt + width/2;
  t3 = t4 - t34;
  if ((t4-t34) < midpt)
    t3 = midpt;

  /* Rp/Rs at ingress and egress: */
  pi = idepth/sqrt(fabs(idepth));  /* Not to confuse with pi */
  pe = edepth/sqrt(fabs(edepth));

  for(i=0; i<dims[0]; i++){
    /* Before ingress: */
    if (INDd(t,i) < t1){
      INDd(eclipse,i) = 1.0 + islope*(INDd(t,i)-t1) + idepth;
    }
    /* During ingress:                   */
    /* Eq. (1) of Mandel & Agol (2002):  */
    else if (INDd(t,i) < t2){
      z  = -2*pi*(INDd(t,i)-t1)/t12 + 1 + pi;
      k0 = acos(0.5*(pi*pi + z*z - 1)/pi/z);
      k1 = acos(0.5*(1 - pi*pi + z*z)/z);
      INDd(eclipse,i) = 1 + idepth - sign(idepth)/M_PI * (pi*pi*k0 + k1
                                   - sqrt((z*z - 0.25*pow((1+z*z-pi*pi),2))));
    }
    /* Totality:       */
    else if (INDd(t,i) < t3){
      INDd(eclipse,i) = 1.0;
    }
    /* During egress:  */
    else if (INDd(t,i) < t4){
      z  = 2*pe*(INDd(t,i)-t3)/t34 + 1 - pe;
      k0 = acos((pe*pe+z*z-1)/2/pe/z);
      k1 = acos((1-pe*pe+z*z)/2/z);
      INDd(eclipse,i) = 1 + edepth - sign(edepth)/M_PI*(pe*pe*k0 + k1
                                   - sqrt((z*z - 0.25*pow((1+z*z-pe*pe),2))));
    }
    /* After egress:   */
    else if (INDd(t,i) >= t4){
      INDd(eclipse,i) = 1.0 + eslope*(INDd(t,i)-t4) + edepth;
    }
    /* Flux normalization: */
    INDd(eclipse,i) *= flux;
  }
  return Py_BuildValue("N", eclipse);
}



PyDoc_STRVAR(eclipse_quad__doc__,
"Secondary-eclipse light-curve model with independent ingress and\n\
egress depths and quadratic baseline.");

static PyObject *eclipse_quad(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, idepth, edepth, t12, t34, flux, slope, quad;
  double t1, t2, t3, t4, pi, pe, z, k0, k1;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt  = INDd(params, 0);
  width  = INDd(params, 1);
  idepth = INDd(params, 2);  // Ingress depth
  edepth = INDd(params, 3);  // Egress depth
  t12    = INDd(params, 4);
  t34    = INDd(params, 5);
  flux   = INDd(params, 6);
  slope  = INDd(params, 7);
  quad   = INDd(params, 8);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  if(idepth == 0){
      for(i=0; i<dims[0]; i++){
        INDd(eclipse, i) = flux;
      }
      return Py_BuildValue("N", eclipse);
    }

  /* Time of contact points: */
  t1 = midpt - width/2;
  t2 = t1 + t12;
  /* Grazing eclipse:        */
  if ((t1+t12) > midpt)
    t2 = midpt;

  t4 = midpt + width/2;
  t3 = t4 - t34;
  if ((t4-t34) < midpt)
    t3 = midpt;

  /* Rp/Rs at ingress and egress: */
  pi = idepth/sqrt(fabs(idepth));  /* Not to confuse with pi */
  pe = edepth/sqrt(fabs(edepth));

  for(i=0; i<dims[0]; i++){
    /* Before ingress: */
    if (INDd(t,i) < t1){
      INDd(eclipse,i) = 1.0 + quad *(INDd(t,i)*INDd(t,i)-t1*t1)
                            + slope*(INDd(t,i)-t1) + idepth;
    }
    /* During ingress:                   */
    /* Eq. (1) of Mandel & Agol (2002):  */
    else if (INDd(t,i) < t2){
      z  = -2*pi*(INDd(t,i)-t1)/t12 + 1 + pi;
      k0 = acos(0.5*(pi*pi + z*z - 1)/pi/z);
      k1 = acos(0.5*(1 - pi*pi + z*z)/z);
      INDd(eclipse,i) = 1 + idepth - sign(idepth)/M_PI * (pi*pi*k0 + k1
                                   - sqrt((z*z - 0.25*pow((1+z*z-pi*pi),2))));
    }
    /* Totality:       */
    else if (INDd(t,i) < t3){
      INDd(eclipse,i) = 1.0;
    }
    /* During egress:  */
    else if (INDd(t,i) < t4){
      z  = 2*pe*(INDd(t,i)-t3)/t34 + 1 - pe;
      k0 = acos((pe*pe+z*z-1)/2/pe/z);
      k1 = acos((1-pe*pe+z*z)/2/z);
      INDd(eclipse,i) = 1 + edepth - sign(edepth)/M_PI*(pe*pe*k0 + k1
                                   - sqrt((z*z - 0.25*pow((1+z*z-pe*pe),2))));
    }
    /* After egress:   */
    else if (INDd(t,i) >= t4){
      INDd(eclipse,i) = 1.0 + quad *(INDd(t,i)*INDd(t,i) - t4*t4)
                            + slope*(INDd(t,i)-t4) + edepth;
    }
    /* Flux normalization: */
    INDd(eclipse,i) *= flux;
  }
  return Py_BuildValue("N", eclipse);
}


PyDoc_STRVAR(eclipse__doc__,
             "Eclipse light-curve models for the Webb.\n");


static PyMethodDef eclipse_methods[] = {
  {"mandelecl",    mandelecl,    METH_VARARGS, mandelecl__doc__},
  {"eclipse_flat", eclipse_flat, METH_VARARGS, eclipse_flat__doc__},
  {"eclipse_lin",  eclipse_lin,  METH_VARARGS, eclipse_lin__doc__},
  {"eclipse_quad", eclipse_quad, METH_VARARGS, eclipse_quad__doc__},
  {NULL,           NULL,         0,            NULL}
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "_eclipse", eclipse__doc__, -1, eclipse_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__eclipse (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_eclipse(void){
  Py_InitModule3("_eclipse", eclipse_methods, eclipse__doc__);
  import_array();
}
#endif
