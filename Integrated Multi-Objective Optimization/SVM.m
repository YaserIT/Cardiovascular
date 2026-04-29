function [Yp, scores]=SVM(train,l_train,test)
        mdl =fitcsvm(train,l_train);
        [Yp, scores] = predict(mdl,test);
end
