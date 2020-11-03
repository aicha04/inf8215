include "globals.mzn"; 

int: n;
array[1..n, 1..n] of var 0..1: grid; % var is used in order to take into account the "not-bounded case" _
 
% decision variables: mine or not on the specific cell
array[1..n, 1..n] of var 0..1: x;


% initializing the grid: '_' values on the grid correspond to variables not already set (i.e. the choices you have to make)
constraint forall(i in 1..n, j in 1..n)(x[i,j] = grid[i,j]);

% For each row and column, same number of cells with and without mine
constraint forall(i in 1..n)(count(j in 1..n)(x[i,j] = 1) = count(j in 1..n)(x[i,j] = 0));
constraint forall(i in 1..n)(count(j in 1..n)(x[j,i] = 1) = count(j in 1..n)(x[j,i] = 0));

% For each row and column, no more than 2 consecutive cells with mine
constraint forall(i in 1..n)(forall(j in 1..n-2)(sum(k in 0..2)(x[i,j+k]) != 3));
constraint forall(i in 1..n)(forall(j in 1..n-2)(sum(k in 0..2)(x[j+k,i]) != 3));

% For each row and column, no more than 2 consecutive cells without mine
constraint forall(i in 1..n)(forall(j in 1..n-2)(sum(k in 0..2)(x[i,j+k]) != 0));
constraint forall(i in 1..n)(forall(j in 1..n-2)(sum(k in 0..2)(x[j+k,i]) != 0));
 
output [
   if j = 1 then "\n" else " " endif ++
     show(x[i,j])
   | i,j in 1..n
];