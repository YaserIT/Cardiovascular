function [Yp,scores]=NB(train,train_lab,test)
model = fitcnb(train, train_lab,'DistributionNames','kernel');
[Yp,scores]=predict(model,test);
end