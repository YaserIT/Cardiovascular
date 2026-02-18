%-------------------------------------------------------------------------%
%  Fitness Function (Error Rate) source codes demo version                %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%  

function fitness=jFitnessFunction(feat,label,X,opts)
w1=0.99;
w2=0.01;
if sum(X==1)==0
  fitness=inf;
else
    f1=jwrapperKNN(feat(:,X==1),label,opts);
    f2=length(nonzeros(X))/size(feat,2);
  fitness=w1*f1+ w2*f2;
end
end


function ER=jwrapperKNN(feat,label,opts)
%---// Parameter setting for k-value of KNN //
k=opts.k; 
%---// Parameter setting for hold-out (20% for testing set) //
ho=0.2;
Model=fitcknn(feat,label,'NumNeighbors',k,'Distance','euclidean'); 
C=crossval(Model,'holdout',ho);
ER=kfoldLoss(C);
end







