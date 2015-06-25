function [rv] = random_vec(n)

i = 0;
rng(0,'twister');
rv = [];

while i < n
    
    rfactor = 2.*rand(1,1)-1;
    
    x = 2.*rand(1,1)-1;
    y = 2.*rand(1,1)-1-x;
    z = sqrt(1-(x*x)-(y*y));
    
    if isreal(z)
        if sign(rfactor) > 0
            rv = vertcat(rv,[x y z]);
        else
            rv = vertcat(rv,[-x -y -z]);
        end
    else
        n = n+1;
    end
    i = i + 1;
end