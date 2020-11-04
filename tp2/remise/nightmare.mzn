include "globals.mzn"; 
   
% Parameters of the problems
int: n = 5;
array[1..n] of int: left_vision = [2,1,3,2,3];
array[1..n] of int: right_vision = [3,4,1,3,2];
array[1..n] of int: up_vision = [2,1,3,2,2];
array[1..n] of int: down_vision = [3,4,2,1,2];

% Variable: value for each cell
array[1..n,1..n] of var 1..n: city;

% For each row and column, no tower with same size
constraint forall(i in 1..n)(all_different([city[i,j] | j in 1..n]));
constraint forall(i in 1..n)(all_different([city[j,i] | j in 1..n]));

constraint forall(i in 1..n)(count(j in 1..n)(count(k in 1..n where k<j)(city[i,j]>city[i,k])= j-1)=up_vision[i]);
constraint forall(i in 1..n)(count(j in 1..n)(count(k in 1..n where k>j)(city[i,j]>city[i,k])= 5-j)=down_vision[i]);
constraint forall(j in 1..n)(count(i in 1..n)(count(k in 1..n where k<i)(city[i,j]>city[k,j])= i-1)=left_vision[j]);
constraint forall(j in 1..n)(count(i in 1..n)(count(k in 1..n where k>i)(city[i,j]>city[k,j])= 5-i)=right_vision[j]);

solve satisfy; 

output
["city:"]++
[
  if j = 1 then "\n" else " " endif ++
    show(city[j,i])
  | i in 1..n, j in 1..n
];
