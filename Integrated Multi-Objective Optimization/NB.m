function Yp=NB(train,train_lab,test)
model = fitctree(train, train_lab);
Yp=predict(model,test);
end