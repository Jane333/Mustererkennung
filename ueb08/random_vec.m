function [rv] = random_vec(n)

i = 0;
rng(0,'twister');
rv = [];

while i < n
    
    rfactor = 2.*rand(1,1)-1;
    
    x = 2.*rand(1,1)-1;
    y = 2.*rand(1,1)-1;
    z = 2.*rand(1,1)-1;
    
    if isreal(z)
        if sign(rfactor) > 0
            uv = [x y z];
            uv = uv / norm(uv);
            rv = vertcat(rv,uv);
        else
            uv = [-x -y -z];
            uv = uv / norm(uv);
            rv = vertcat(rv,uv);
        end
    else
        n = n+1;
    end
    i = i + 1;
end