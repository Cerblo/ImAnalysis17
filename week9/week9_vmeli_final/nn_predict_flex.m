function zfin = nn_predict_flex(x,weights)

[n,~]=size(x);

nb_hl=weights(end).layer;

weights(1).z=x;


    for l=1:nb_hl-1
           
        
            weights(l+1).a =weights(l).z*weights(l+1).w+repmat(weights(l).b,n,1); 
            weights(l+1).z = max( weights(l+1).a,0);
    end  
            
            weights(nb_hl+1).a =weights(nb_hl).z*weights(nb_hl+1).w+repmat(weights(nb_hl).b,n,1); 
            weights(nb_hl+1).z = softmax_func( weights(nb_hl+1).a);
            
            zfin=weights(nb_hl+1).z;
end
