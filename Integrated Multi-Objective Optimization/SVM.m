function Yp=SVM(train,l_train,test)
    for i=1:size(l_train,2)
        mdl =fitcsvm(train,l_train(:,i));
        Yp(i,:) = predict(mdl,test);
    end
    Yp=Yp';
end
