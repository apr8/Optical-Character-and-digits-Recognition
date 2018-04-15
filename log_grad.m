function G=log_grad(y, X, B) 

%compute gradient 

    K_1=size(B,2);
    eXB=exp(X*B);
    s_eXB=1./(sum(eXB, 2)+1);
    eXB=eXB.*repmat(s_eXB, 1, K_1);
    
    for k= 1: K_1
        eXB(:,k)=(y==k)-eXB(:,k);
    end
    
    G=(X'*eXB);

end