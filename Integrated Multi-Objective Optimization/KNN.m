function [Yp, score]=KNN(train,l_train,test)
mdl=fitcknn(train,l_train);
[Yp,score] = predict(mdl,test);
end
