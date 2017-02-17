function  [edgeIm] = getEdge(im, t)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[g, dG, ddG, dddG] = getGaussFilt(t);

xMax=round(5*sqrt(t));
x=-xMax:xMax;

Lx = filter2(dG(x), im);
Lx = filter2(g(x)', Lx);

Lxx=filter2(ddG(x), im);
Lxx = filter2(g(x)', Lxx);


Ly = filter2(dG(x)', im);
Ly = filter2(g(x), Ly);

Lyy=filter2(ddG(x)', im);
Lyy = filter2(g(x), Lyy);
% 
Lxy=filter2(dG(x), im);
Lxy=filter2(dG(x)', Lxy);

%

Lxxx= filter2(dddG(x),im);
Lxxx= filter2(g(x)', Lxxx);

Lyyy= filter2(dddG(x)',im);
Lyyy= filter2(g(x), Lyyy);

Lxxy= filter2(ddG(x),im);
Lxxy= filter2(dG(x)', Lxxy);

Lxyy= filter2(ddG(x)',im);
Lxyy= filter2(g(x), Lxyy);

Lvv=Lx.^2.*Lxx+2*Lx.*Ly.*Lxy+Ly.^2.*Lyy;
Lvvv=Lx.^3.*Lxxx+3*Lx.^2.*Ly.*Lxxy+3*Lx.*Ly.^2.*Lxyy+Ly.^3.*Lyyy;



LvvvP = ( Lvvv < 0 );




LvvP = ( Lvv > 0 );
LvvP = ( xor( LvvP, circshift( LvvP, [0,1] ) ) |...
         xor( LvvP, circshift( LvvP, [1,0] ) ) );
edgeIm=LvvP.*LvvvP;

       
end

