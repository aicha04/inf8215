include "globals.mzn";

int: n_object;
array[1..n_object,1..2] of int: shape; % (x,y) shape of each object
int: max_length;
int: max_width;

array[1..n_object,1..2] of var 1..max(max_width,max_length): coords;
var int: width;
var int: length;

function var int: mesure(int: i, int:coord) = coords[i,coord]+shape[i,coord]-1;

constraint forall(i in 1..n_object)(mesure(i,1) <= max_length);
constraint forall(i in 1..n_object)(mesure(i,2) <= max_width);
constraint length = sum(i in 1..n_object)(shape[i,1]);
constraint width = max([mesure(i,2) | i in 1..n_object]);
constraint forall(i in 1..n_object)(coords[i,1] = sum(j in 1..i)(shape[j,1]));

solve satisfy;

output ["length = \(length)\n"] ++
["width = \(width)\n"] ++
["area = \(length*width)"] ++
[
   if j = 1 then "\n" else " " endif ++
     show(coords[i,j])
   | i in 1..n_object,j in 1..2
];
