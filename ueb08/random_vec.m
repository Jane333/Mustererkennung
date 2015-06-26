function [rv] = random_vec(n)

rng(0,'twister');
rv = [];

for i = 1:n
    
    varX = 2*pi*rand();
    varY = acos(2*rand()-1);
    
    x = cos(varX) * sin(varY);
    y = sin(varX) * sin(varY);
    z = 1         * cos(varY);
   
    vec = [x y z];
    rv = vertcat(rv,vec);

end