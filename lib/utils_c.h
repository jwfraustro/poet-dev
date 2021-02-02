/*
Definitions:
 define bool type

Utility functions:
 MAX : return largest  value between two inputs
 MIN : return smallest value between two inputs
 SWAP: swap two values
 matrix_doub: create a pointer-to-pointers, emulating a 2D array (double)
 matrix_char: create a pointer-to-pointers, emulating a 2D array (char)

Modification history:
2012-04-24  patricio  First implementation
*/

typedef char bool;

/*inline*/ int MAX(const int a, const int b){
  return b > a ? b : a;
}

/*inline*/ int MIN(const int a, const int b){
  return b < a ? b : a;
}

void SWAP(double *i, double *j) {
  double t = *i;
  *i = *j;
  *j = t;
  return;
}

/*
void matrix_doub(double **matrix, double *vector; int nx, int ny){
  // make a pointer to pointers of doubles to emulate a matrix
  int i;
  vector = malloc(nx * ny * sizeof(double));
  matrix = malloc(nx * sizeof(double *));
  for (i = 0; i < nx; i++)
    matrix[i] = vector + (i * ny);
  return;
}

void matrix_char(char **matrix, char *vector, int nx, int ny){
  // make a pointer to pointers of doubles to emulate a matrix
  int i;
  vector = malloc(nx * ny * sizeof(char));
  matrix = malloc(nx * sizeof(char *));
  for (i = 0; i < nx; i++)
    matrix[i] = vector + (i * ny);
  return;
}
*/
