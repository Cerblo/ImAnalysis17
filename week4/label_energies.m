function [LE] = label_energies(S,D,mu,alpha)
nr=size(D,1);
nc=size(D,2);
LE=zeros(nr, nc, length(mu));



for k=1:length(mu)
    for x = 1:size(S,2)
        for y = 1:size(S,1)
            likelihood = alpha * (mu(k) - D(x,y))^2;
            prior = ~(x ~= 1 && S(x-1,y) == k) + ~(x~= 100 && S(x+1,y) == k) ...
                + ~(y ~= 1 && S(x,y-1) == k) + ~(y ~= 100 && S(x,y+1) == k);
            LE(x,y,k) = likelihood + prior;
        end
    end
end
end

