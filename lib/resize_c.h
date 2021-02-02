/*
Main functions:
 resize:      resize a 2D data image using bi-linear interpolation.
 resize_mask: resize a 2D mask image.

Sub-routines:
 locate: 
 hunt:
 poly_interp:

Modification history:
2012-04-24  patricio  First implementation
*/

int locate(const double x, int n, int mm, double *xx, int jsav,
	   int cor, int dj){
  // return value j such x is centered in the range xx[j..j+mm-1]
  // from numerical recipes
  int ju, jm, jl;
  if (n < 2 || mm < 2 || mm > n){
    printf("locate size error");
    exit(-1);
  }
  // is the array in ascending order?
  bool ascnd = (xx[n-1] >= xx[0]);
  jl = 0;    // initialize lower limit
  ju = n-1;  // initialize upper limit
  // Bisection search:
  while (ju-jl > 1){     
    jm = (ju+jl) >> 1;  // compute midpoint
    if ( (x >= xx[jm]) == ascnd)
      jl=jm;            // replace limits
    else
      ju=jm;
  }
  cor = abs(jl-jsav) > dj ? 0 : 1;
  jsav = jl;                         
  return MAX(0, MIN(n-mm, jl-((mm-2)>>1)));
}


int hunt(const double x, int n, int mm, double *xx, int jsav, int cor, int dj){
  int jl=jsav, jm, ju, inc=1;
  // from numerical recipes
  if (n < 2 || mm < 2 || mm > n){ 
    printf("hunt size error");
    exit(-1);
  }
  // is the array in ascending order?
  bool ascnd=(xx[n-1] >= xx[0]);
  if (jl < 0 || jl > n-1) {
    jl=0;
    ju=n-1;
  } else {
    if ( (x >= xx[jl]) == ascnd) {
      for (;;) {
	ju = jl + inc;
	if (ju >= n-1) { ju = n-1; break;}
	else if ( (x < xx[ju]) == ascnd) break;
	else {
	  jl = ju;
	  inc += inc;
	}
      }
    } else {
      ju = jl;
      for (;;) {
	jl = jl - inc;
	if (jl <= 0) { jl = 0; break;}
	else if ( (x >= xx[jl]) == ascnd) break;
	else {
	  ju = jl;
	  inc += inc;
	}
      }
    }
  }
  while (ju-jl > 1) {
    jm = (ju+jl) >> 1;
    if ( (x >= xx[jm]) == ascnd)
      jl=jm;
    else
      ju=jm;
  }
  cor = abs(jl-jsav) > dj ? 0 : 1;
  jsav = jl;
  return MAX(0, MIN(n-mm, jl-((mm-2)>>1)));
}


double poly_interp(double x, int n, int mm, double *xx, double *yy, 
		   int jsav, int cor, int dj){
  // from numerical recipes
  int jl = cor ? hunt(  x, n, mm, xx, jsav, cor, dj) : 
                 locate(x, n, mm, xx, jsav, cor, dj);

  // do the interpolation: but how?
  double dy = 0.0;
  int i, m, ns=0;
  double y, den, dif, dift, ho, hp, w;
  const double *xa = &xx[jl], *ya = &yy[jl];
  double c[mm], d[mm];
  dif = abs(x-xa[0]);
  for (i=0; i<mm; i++) {
    if ((dift=abs(x-xa[i])) < dif) {
      ns  = i;
      dif = dift;
    }
    c[i] = ya[i];
    d[i] = ya[i];
  }
  y = ya[ns--];
  for   (m=1; m<mm;   m++) {
    for (i=0; i<mm-m; i++) {
      ho = xa[i]-x;
      hp = xa[i+m]-x;
      w  = c[i+1]-d[i];
      if ((den=ho-hp) == 0.0){
	printf("Poly_interp error");
	exit(0);
      }
      den  = w/den;
      d[i] = hp*den;
      c[i] = ho*den;
    }
    y += (dy=(2*(ns+1) < (mm-m) ? c[ns+1] : d[ns--]));
  }
  return y;
}


void resize(double **idata, double **data, int expand, int nx, int ny, 
            int mx, int my){
  // resize a 2D data image using bi-linear interpolation.
  // store result in idata.
  int i, j, n, m;
  double x[nx], y[ny], xx[mx], yy[my];
  double t, u;
  int degree = 2;

  if (expand==1){ // no resize
    for   (i=0; i<nx; i++)
      for (j=0; j<ny; j++)
	idata[i][j] = data[i][j];
    return;
  }

  int jsavx=0, jsavy=0; 
  int corx=0,  cory=0;
  int djx = MIN(1, (int)pow((double)nx,0.25));
  int djy = MIN(1, (int)pow((double)ny,0.25));

  for (i=0; i<nx; i++)
    x[i] = i;

  for (j=0; j<ny; j++)
    y[j] = j;

  for (i=0; i<mx; i++)
    xx[i] = i*1.0/expand;

  for (j=0; j<my; j++)
    yy[j] = j*1.0/expand;

  for   (n=0; n<mx; n++){
    i = corx ? hunt(  xx[n], nx, degree, x, jsavx, corx, djx) : 
               locate(xx[n], nx, degree, x, jsavx, corx, djx);
    t = (xx[n]-x[i]) / (x[i+1]-x[i]);
    for (m=0; m<my; m++){
      j = cory ? hunt(  yy[m], ny, degree, y, jsavy, cory, djy) : 
                 locate(yy[m], ny, degree, y, jsavy, cory, djy);
      u = (yy[m]-y[j]) / (y[j+1]-y[j]);
      // interpolated value
      idata[n][m] = (1.0-t)*(1.0-u)*data[i][j  ] + t*      u*data[i+1][j+1] + 
                    (1.0-t)*      u*data[i][j+1] + t*(1.0-u)*data[i+1][j  ];
    }
  }
  return;
}


void resize_mask(char **idata, double **data, int expand, int nx, int ny, 
		 int mx, int my, double **rdata){
  // resize for a mask array (char type), return 0 if value < 1.0
  int i, j;

  // store in rdata the interpolated values (of double type)
  resize(rdata, data, expand, nx, ny, mx, my);
  for (i=0; i<mx; i++)
    for (j=0; j<my; j++)
      idata[i][j] = (char)((int)(round(rdata[i][j]*100.0))/100);
  return;
}
