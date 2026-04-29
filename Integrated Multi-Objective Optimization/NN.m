function [label,scores]=NN(trn,l_train,test)
    mdl=fitcnet(trn,l_train);
    [label,scores]= predict(mdl,test);
end