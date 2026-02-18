function Yp=NB(train,train_lab,test)
NBmodel=fitcnb(train,train_lab);
Yp=predict(NBmodel,test);
end