function [err,weigths]= nn_train_flex( x, t, lrate,nb_iter, nb_neurons )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%% INPUT
%M: number of features, N: number of obs, P:number of targets attributes,
%L: number of hidden layers
%x:features, size:NXM
%t:target, size: NXP
%nb iter
%h:structure with all the activation functions
%nb_neurons: size 1XL
%% OUTPUT
%w: struct with all the weights for each layer
%bias: for each layer

[n,m] = size(x);
nb_hl=length(nb_neurons)+1; %because we define only the intern nb of neurons
[~,p] = size(t);
nb_neurons_full=[m,nb_neurons,p];

weigths=struct('layer',{},'w',{},'z',{},'a',{},'delta',{},'b',{} );
E=zeros(nb_iter,1);
weigths(1).w={};
weigths(1).a={};
weigths(1).delta={};
weigths(1).z=x;
weigths(nb_hl+1).b={};
weigths(1).layer=0;

err=zeros(nb_iter,1);

%random init
for l=1:nb_hl
            weigths(l+1).layer=l;
            weigths(l+1).w=1/(nb_neurons_full(1,l))*rand(nb_neurons_full(1,l),nb_neurons_full(1,l+1));
            weigths(l).b=1/(nb_neurons_full(1,l))*rand(1,nb_neurons_full(1,l+1));
end

for iter=1:nb_iter
   
   
    
    for l=1:nb_hl-1
            
            weigths(l+1).a =weigths(l).z*weigths(l+1).w+repmat(weigths(l).b,n,1); 
            weigths(l+1).z = max( weigths(l+1).a,0);
    end  
           
            weigths(nb_hl+1).a =weigths(nb_hl).z*weigths(nb_hl+1).w+repmat(weigths(nb_hl).b,n,1); 
            weigths(nb_hl+1).z = softmax_func( weigths(nb_hl+1).a);
            
    %Energy comp
    
    weigths(nb_hl+1).delta = weigths(nb_hl+1).z - t;
    weigths(nb_hl+1).w = weigths(nb_hl+1).w - lrate*1/n*weigths(nb_hl).z'*weigths(nb_hl+1).delta ;
    
    for l=1:nb_hl-1
        
        weigths(nb_hl-l+1).delta =max(0, weigths(nb_hl-l+1).a ./abs(weigths(nb_hl-l+1).a)).*(weigths(nb_hl-l+2).delta*weigths(nb_hl-l+2).w') ;
        weigths(nb_hl-l+1).b = weigths(nb_hl-l+1).b - lrate *mean(weigths(nb_hl-l+2).delta,1) ;  
        weigths(nb_hl-l+1).w = weigths(nb_hl-l+1).w - lrate*1/n*weigths(nb_hl-l).z'*weigths(nb_hl-l+1).delta ;
    end 

     weigths(1).b = weigths(1).b - lrate *mean(weigths(2).delta,1) ;  
    
    z_true=weigths(nb_hl+1).z;
    probs = z_true(:,1)>= 0.5;
    y_est=[probs,~probs];
    
    err(iter,1)=norm(y_est-t)/sqrt(n)*100;

   
end



end

