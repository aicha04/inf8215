include "globals.mzn";

int: n; 
array [1..n] of var 1..n: d; % One variable for each column. The domain corresponds to the cell in the column where the drone must be located

constraint all_different(d);
constraint all_different([ d[i] + i | i in 1..n ]); % distinct diagonals
constraint all_different([ d[i] - i | i in 1..n ]); % upwards+downwards

solve satisfy;

output	
  [if fix(d[i]) = j then "D " 
   else ". " 
   endif
   
   ++
   
   if j = n then "\n" 
   else "" endif	
   | i, j in 1..n];
   