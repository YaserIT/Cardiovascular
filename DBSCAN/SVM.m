function Yp = SVM(train,l_train,test)
    svmStruct = fitcsvm(train,l_train);
    Yp = predict(svmStruct,test);
end