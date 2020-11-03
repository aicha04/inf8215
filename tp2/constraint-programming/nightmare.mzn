
include "globals.mzn"; 
   
% Parameters of the problems
int: n = 5;
array[1..n] of int: left_vision = [2,1,3,2,3];
array[1..n] of int: right_vision = [3,4,1,3,2];
array[1..n] of int: up_vision = [2,1,3,2,2];
array[1..n] of int: down_vision = [3,4,2,1,2];

% Variable: value for each cell
array[1..n,1..n] of var 1..n: city;

constraint forall(i in 1..n)(all_different([city[i,j] | j in 1..n]));
constraint forall(i in 1..n)(all_different([city[j,i] | j in 1..n]));

constraint forall(i in 1..n)((count(j in 1..n-1)(city[i,j]<city[i,j+1]) = left_vision[i]));
constraint forall(i in 1..n)((count(j in 1..n-1)(city[i,j]>city[i,j+1]) = right_vision[i]));

output
["city:"]++
[
  if j = 1 then "\n" else " " endif ++
    show(city[j,i])
  | i in 1..n, j in 1..n
];
