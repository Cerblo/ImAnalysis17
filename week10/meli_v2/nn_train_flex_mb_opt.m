function [err,err_ep,weigths]= nn_train_flex_mb_opt( x, t, lrate,nb_iter, nb_neurons,mb_size )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%% INPUT
%M: number of features, N: number of obs, P:number of targets attributes,
%L: number of hidden layers
%x:features, size:NXM
%t:target, size: NXP
%nb iter
%nb_neurons: size 1XL
%mb_size: size of the minibatch
%% OUTPUT
%weigths: struct with all the weights for each layer

delt=10^(-7);
[n,m] = size(x);
nb_hl=length(nb_neurons)+1; %because we define only the intern nb of neurons
[~,p] = size(t);
nb_neurons_full=[m,nb_neurons,p];

%build the structure containg infos on ANN
weigths=struct('layer',{},'w',{},'z',{},'a',{},'delta',{},'b',{},'r',{},'rb',{} );


weigths(1).w={};
weigths(1).a={};
weigths(1).delta={};
weigths(nb_hl+1).b={};
weigths(1).layer=0;
weigths(1).r={};
weigths(nb_hl+1).rb={};

% definining errors
err=zeros(nb_iter,1);%For each iteration
err_ep=zeros(floor(nb_iter/n),1);%for each epoch



%random init
for l=1:nb_hl
            weigths(l+1).layer=l;
            weigths(l+1).r=zeros(nb_neurons_full(1,l),nb_neurons_full(1,l+1));
            weigths(l).rb=zeros(1,nb_neurons_full(1,l+1));
            weigths(l+1).w=1/(nb_neurons_full(1,l))*rand(nb_neurons_full(1,l),nb_neurons_full(1,l+1));
            weigths(l).b=1/(nb_neurons_full(1,l))*rand(1,nb_neurons_full(1,l+1));
end

iter_data=1;
iter_err_ep=1;

i_beg=1;

for iter=1:nb_iter
%     idx=randsample(n,mb_size);
%     weigths(1).z=x(idx,:);
%     targ=t(idx,:);
%     

%Going linearly through all the dataset, building epochs
    weigths(1).z=x(1+(iter_data-1)*mb_size:iter_data*mb_size,:);
    
    targ=t(1+(iter_data-1)*mb_size:iter_data*mb_size,:);
    if iter_data*mb_size>=n
        
        %computation of error per epoch
        err_ep(iter_err_ep,1)=mean(err(i_beg:iter-1,1));
        i_beg=iter;
        iter_err_ep=iter_err_ep+1;  
        
       iter_data=1;
        
    else
        iter_data=iter_data+1;
    end
    %Forward prop
    
    for l=1:nb_hl-1
            
            weigths(l+1).a =weigths(l).z*weigths(l+1).w+repmat(weigths(l).b,mb_size,1); 
            weigths(l+1).z = max( weigths(l+1).a,0);
              
    end  
           
            weigths(nb_hl+1).a =weigths(nb_hl).z*weigths(nb_hl+1).w+repmat(weigths(nb_hl).b,mb_size,1); 
            weigths(nb_hl+1).z = softmax_func( weigths(nb_hl+1).a);
           
    %Backpropag
    
    weigths(nb_hl+1).delta = weigths(nb_hl+1).z - targ;
    g=1/mb_size *weigths(nb_hl).z'*weigths(nb_hl+1).delta;
    weigths(nb_hl+1).r=weigths(nb_hl+1).r+g.*g;
 
    eps=lrate./(delt+(weigths(nb_hl+1).r).^(1/2));
    weigths(nb_hl+1).w = weigths(nb_hl+1).w -eps.*g;
   
    for l=1:nb_hl-1
        
        weigths(nb_hl-l+1).delta =max(0, weigths(nb_hl-l+1).a ./abs(weigths(nb_hl-l+1).a)).*(weigths(nb_hl-l+2).delta*weigths(nb_hl-l+2).w') ;
        g=1/mb_size *weigths(nb_hl-l).z'*weigths(nb_hl-l+1).delta ;
        weigths(nb_hl-l+1).r= weigths(nb_hl-l+1).r+g.*g;
        eps=lrate./(delt+(weigths(nb_hl-l+1).r).^(1/2));
        weigths(nb_hl-l+1).w = weigths(nb_hl-l+1).w -eps.*g;
        
        
        gb=mean(weigths(nb_hl-l+2).delta,1) ;
        weigths(nb_hl-l+1).rb =weigths(nb_hl-l+1).rb+gb.*gb;
        epsb=lrate./(delt+(weigths(nb_hl-l+1).rb).^(1/2));
        weigths(nb_hl-l+1).b = weigths(nb_hl-l+1).b -epsb.*gb;  

    end 
    
    gb=mean(weigths(2).delta,1);
    weigths(1).rb =weigths(1).rb+gb.*gb;
    epsb=lrate./(delt+(weigths(1).rb).^(1/2));  
    weigths(1).b = weigths(1).b - -epsb.*gb ;  
   
    %Estimate the error
    z_true=weigths(nb_hl+1).z;
    [max_y_e,ind_ye] = max(z_true,[],2);
    [max_y_test, ind_y_test]=max(targ,[],2);
    
    err(iter,1)=sum(ind_ye~=ind_y_test)/mb_size*100;
end



end


