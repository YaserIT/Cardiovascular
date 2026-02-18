clear
clc
close all
load data
%%
D=[cleveland;Hungarian;switzerland;VA];
C=[clevelandClass;HungarianClass;switzerlandClass;VAClass];
%%
[m n]=size(D);
C1=0;C2=0;
for i=1:m
    for j=1:n
        if(D(i,j)==-9)
            D(i,j)=0;
        elseif isnan(D(i,j))
            D(i,j)=0;            
        elseif isnan(C(i,1))
            C(i,1)=0;
        end
    end
    if C(i,1)==0
        C1=C1+1;
        class(i,1)=1;
        Cluster1(C1,:)=D(i,:);
    else
        C2=C2+1;
        class(i,1)=2;
        Cluster2(C2,:)=D(i,:);
    end
end
%%
Cen1=Cluster1(randi(C1),:);
for i=1:length(Cluster1)
    for j=1:n
        H1(i,:)=((Cluster1(i,j)-Cen1(1,j))^2);
    end
    Dist1(i,1)=sqrt(sum(H1));
end
Cen2=Cluster2(randi(C2),:);
for i=1:length(Cluster2)
    for j=1:n
    H2(i,:)=((Cluster2(i,j)-Cen2(1,j))^2);
    end
    Dist2(i,2)=sum(sqrt(H2));
end
%%
S=sortrows([std(D) ; 1:n]',-1);
FS=S(1:3,:);
%%
g=5;
h=8;
k=9;
figure
scatter3(Cluster1(:,g),Cluster1(:,h),Cluster1(:,k),'r')
hold on
scatter3(Cluster2(:,g),Cluster2(:,h),Cluster2(:,k),'k')
hold off
pause (0.01)
xlabel('heart beat rate')
ylabel( 'blood pressure')
zlabel('exercise induced angina')
legend('Normal', 'Heart patient', 'Location','northoutside','Orientation','horizontal')
%%
X=[D(:,5),D(:,8),D(:,9)];
%%
for i = 1:100:m
p=length(find(C(1:i)==class(1)));
epsilon(i,1)=round((p/m)*100);
MinPts(i,1)=round((p/(m-p))*50);
index=DBSCAN(X,epsilon(i,1),MinPts(i,1));
% Plot Results
f=figure
PlotClusterinResult(X, index);
title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon(i,1)) ', MinPts = ' num2str(MinPts(i,1)) ')']);

improv(i,1)= length(unique(index))/m;
end
%%
figure
plot(nonzeros(improv),'.-b','LineWidth',2,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',15)
xlabel('Itration')
xlim([1 length(nonzeros(improv))])
ylabel('optimal point')
ylim([0 1])
title('Convergence to the optimal point')
%%
for i=1:length(unique(class))
    p=length(find(C==class(i)));
epsilon(i,1)=round((p/m)*100);
MinPts(i,1)=round((p/(m-p))*50);
IDX(:,i)=DBSCAN(X(:,1:2),epsilon(i,1),MinPts(i,1));
% Plot Results
f=figure
PlotClusterinResult(X, IDX(:,i));
title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon(i,1)) ', MinPts = ' num2str(MinPts(i,1)) ')']);
% keyboard
end
%%
uni = unique(IDX(:,1));
for j=1:length(uni)
    Count(j,1)=length(find(IDX(:,1)==uni(j)));
    if uni(j)==0
    disp(['The number of anomalies is ', num2str(Count(j,1)), ' that is ', num2str((Count(j,1)/length(IDX))*100),' percent of total instances']);
    else
    disp(['The number of instances in cluster ', num2str(uni(j)),' is ', num2str(Count(j,1)), ' that is ', num2str((Count(j,1)/length(IDX))*100),' percent of total instances']);
    end
end
%%
tp=0;tn=0;fp=0;fn=0;CO=0;
for i=1:m
    if IDX(i,2)==0 
        if Class(i,1)==2
            tn=tn+1;
        elseif Class(i,1)==1
            fn=fn+1;
        end
    elseif IDX(i,2)==1
        if Class(i,1)==1
            tp=tp+1;
        else
            fp=fp+1;
        end
    elseif IDX(i,2)==2
        if Class(i,1)==1
            fn=fn+1;
        else
            tn=tn+1;
        end
    else
        if Class(i,1)==1
            fn=fn+1;
        else
            tn=tn+1;
        end
    end
      if mod(i,20)==0
        CO=CO+1;
        TP(CO,1)=tp;
         if isnan(TP)
               TP(CO,1)=0;
         end
        TN(CO,1)=tn;
         if isnan(TN)
               TN(CO,1)=0;
         end
        FP(CO,1)=fp;
         if isnan(FP)
               FP(CO,1)=0;
         end
        FN(CO,1)=fn;
        if isnan(FN)
            FN(CO,1)=0;
         end
      end
end
%%
for i=1 :length(TP)
Accuracy(i,1)=(TP(i,1)+TN(i,1))/(TP(i,1)+TN(i,1)+FP(i,1)+FN(i,1));
TruePositiveRate (i,1)= TP(i,1)/(TP(i,1)+FN(i,1));
TrueNegativeRate (i,1)= TN(i,1)/(TN(i,1)+FP(i,1));
Recall(i,1)= TP(i,1)/(TP(i,1)+FN(i,1));
Precision(i,1)= TP(i,1)/(TP(i,1)+FP(i,1));
Fmeasure(i,1)=2/((1/Precision(i,1))+(1/Recall(i,1)));
end

%%
knn=KNN(D,clas,D);
knntp=0;knntn=0;knnfp=0;knnfn=0;knnCO=0;
for i=1:m
    if knn(i,1)==2 
        if class(i,1)==2
            knntp=knntp+1;
        elseif class(i,1)==1
            knnfp=knnfp+1;
        end
   elseif knn(i,1)==1
        if class(i,1)==2
            knnfn=knnfn+1;
        elseif class(i,1)==1
            knntn=knntn+1;
        end
    end
      if mod(i,20)==0
        knnCO=knnCO+1;
        knnTP(knnCO,1)=knntp;
         if isnan(knnTP)
            knnTP(knnCO,1)=0;
         end
             knnTN(knnCO,1)=knntn;
         if isnan(knnTN)
               knnTN(knnCO,1)=0;
         end
        knnFP(knnCO,1)=knnfp;
         if isnan(knnFP)
               knnFP(knnCO,1)=0;
         end
        knnFN(knnCO,1)=knnfn;
        if isnan(knnFN)
            knnFN(knnCO,1)=0;
         end
      end
end
for i=1 :length(knnTP)
knnAccuracy(i,1)=(knnTP(i,1)+knnTN(i,1))/(knnTP(i,1)+knnTN(i,1)+knnFP(i,1)+knnFN(i,1));
knnTruePositiveRate (i,1)= knnTP(i,1)/(knnTP(i,1)+knnFN(i,1));
knnTrueNegativeRate (i,1)= knnTN(i,1)/(knnTN(i,1)+knnFP(i,1));
knnRecall(i,1)= knnTP(i,1)/(knnTP(i,1)+knnFN(i,1));
knnPrecision(i,1)= knnTP(i,1)/(knnTP(i,1)+knnFP(i,1));
knnFmeasure(i,1)=2/((1/knnPrecision(i,1))+(1/knnRecall(i,1)));
end

%% SVM
svm=SVM(D,clas,D);
svmtp=0;svmtn=0;svmfp=0;svmfn=0;svmCO=0;
for i=1:m
    if svm(i,1)==2 
        if class(i,1)==2
            svmtp=svmtp+1;
        elseif class(i,1)==1
            svmfp=svmfp+1;
        end
   elseif svm(i,1)==1
        if class(i,1)==2
            svmfn=svmfn+1;
        elseif class(i,1)==1
            svmtn=svmtn+1;
        end
    end
      if mod(i,20)==0
        svmCO=svmCO+1;
        svmTP(svmCO,1)=svmtp;
         if isnan(svmTP)
            svmTP(svmCO,1)=0;
         end
             svmTN(svmCO,1)=svmtn;
         if isnan(svmTN)
               svmTN(svmCO,1)=0;
         end
        svmFP(svmCO,1)=svmfp;
         if isnan(svmFP)
               svmFP(svmCO,1)=0;
         end
        svmFN(svmCO,1)=svmfn;
        if isnan(svmFN)
            svmFN(svmCO,1)=0;
         end
      end
end
for i=1 :length(svmTP)
svmAccuracy(i,1)=(svmTP(i,1)+svmTN(i,1))/(svmTP(i,1)+svmTN(i,1)+svmFP(i,1)+svmFN(i,1));
svmTruePositiveRate (i,1)= svmTP(i,1)/(svmTP(i,1)+svmFN(i,1));
svmTrueNegativeRate (i,1)= svmTN(i,1)/(svmTN(i,1)+svmFP(i,1));
svmRecall(i,1)= svmTP(i,1)/(svmTP(i,1)+svmFN(i,1));
svmPrecision(i,1)= svmTP(i,1)/(svmTP(i,1)+svmFP(i,1));
svmFmeasure(i,1)=2/((1/svmPrecision(i,1))+(1/svmRecall(i,1)));
end

%% NN
nn=NN(D,clas,D);
nntp=0;nntn=0;nnfp=0;nnfn=0;nnCO=0;
for i=1:m
    if nn(i)==2 
        if class(i,1)==2
            nntp=nntp+1;
        elseif class(i,1)==1
            nnfp=nnfp+1;
        end
   elseif nn(i)==1
        if class(i,1)==2
            nnfn=nnfn+1;
        elseif class(i,1)==1
            nntn=nntn+1;
        end
    end
      if mod(i,20)==0
        nnCO=nnCO+1;
        nnTP(nnCO,1)=nntp;
         if isnan(nnTP)
            nnTP(nnCO,1)=0;
         end
             nnTN(nnCO,1)=nntn;
         if isnan(nnTN)
               nnTN(nnCO,1)=0;
         end
        nnFP(nnCO,1)=nnfp;
         if isnan(nnFP)
               nnFP(nnCO,1)=0;
         end
        nnFN(nnCO,1)=nnfn;
        if isnan(nnFN)
            nnFN(nnCO,1)=0;
         end
      end
end
for i=1 :length(nnTP)
nnAccuracy(i,1)=(nnTP(i,1)+nnTN(i,1))/(nnTP(i,1)+nnTN(i,1)+nnFP(i,1)+nnFN(i,1));
nnTruePositiveRate (i,1)= nnTP(i,1)/(nnTP(i,1)+nnFN(i,1));
nnTrueNegativeRate (i,1)= nnTN(i,1)/(nnTN(i,1)+nnFP(i,1));
nnRecall(i,1)= nnTP(i,1)/(nnTP(i,1)+nnFN(i,1));
nnPrecision(i,1)= nnTP(i,1)/(nnTP(i,1)+nnFP(i,1));
nnFmeasure(i,1)=2/((1/nnPrecision(i,1))+(1/nnRecall(i,1)));
end

%% DT
dt=DT(D,clas,D);
dttp=0;dttn=0;dtfp=0;dtfn=0;dtCO=0;
for i=1:m
    if dt(i,1)==2 
        if class(i,1)==2
            dttp=dttp+1;
        elseif class(i,1)==1
            dtfp=dtfp+1;
        end
   elseif dt(i,1)==1
        if class(i,1)==2
            dtfn=dtfn+1;
        elseif class(i,1)==1
            dttn=dttn+1;
        end
    end
      if mod(i,20)==0
        dtCO=dtCO+1;
        dtTP(dtCO,1)=dttp;
         if isnan(dtTP)
            dtTP(dtCO,1)=0;
         end
             dtTN(dtCO,1)=dttn;
         if isnan(dtTN)
               dtTN(dtCO,1)=0;
         end
        dtFP(dtCO,1)=dtfp;
         if isnan(dtFP)
               dtFP(dtCO,1)=0;
         end
        dtFN(dtCO,1)=dtfn;
        if isnan(dtFN)
            dtFN(dtCO,1)=0;
         end
      end
end
for i=1 :length(nnTP)
dtAccuracy(i,1)=(dtTP(i,1)+dtTN(i,1))/(dtTP(i,1)+dtTN(i,1)+dtFP(i,1)+dtFN(i,1));
dtTruePositiveRate (i,1)= dtTP(i,1)/(dtTP(i,1)+dtFN(i,1));
dtTrueNegativeRate (i,1)= dtTN(i,1)/(dtTN(i,1)+dtFP(i,1));
dtRecall(i,1)= dtTP(i,1)/(dtTP(i,1)+dtFN(i,1));
dtPrecision(i,1)= dtTP(i,1)/(dtTP(i,1)+dtFP(i,1));
dtFmeasure(i,1)=2/((1/dtPrecision(i,1))+(1/dtRecall(i,1)));
end

%% NB
nb=NB(D,clas,D);
nbtp=0;nbtn=0;nbfp=0;nbfn=0;nbCO=0;
for i=1:m
    if nb(i,1)==2 
        if class(i,1)==2
            nbtp=nbtp+1;
        elseif class(i,1)==1
            nbfp=nbfp+1;
        end
   elseif nb(i,1)==1
        if class(i,1)==2
            nbfn=nbfn+1;
        elseif class(i,1)==1
            nbtn=nbtn+1;
        end
    end
      if mod(i,20)==0
        nbCO=nbCO+1;
        nbTP(nbCO,1)=nbtp;
         if isnan(nbTP)
            nbTP(nbCO,1)=0;
         end
             nbTN(nbCO,1)=nbtn;
         if isnan(nbTN)
               nbTN(nbCO,1)=0;
         end
        nbFP(nbCO,1)=nbfp;
         if isnan(nbFP)
               nbFP(nbCO,1)=0;
         end
        nbFN(nbCO,1)=nbfn;
        if isnan(nbFN)
            nbFN(nbCO,1)=0;
         end
      end
end
for i=1 :length(nnTP)
nbAccuracy(i,1)=(nbTP(i,1)+nbTN(i,1))/(nbTP(i,1)+nbTN(i,1)+nbFP(i,1)+nbFN(i,1));
nbTruePositiveRate (i,1)= nbTP(i,1)/(nbTP(i,1)+nbFN(i,1));
nbTrueNegativeRate (i,1)= nbTN(i,1)/(nbTN(i,1)+nbFP(i,1));
nbRecall(i,1)= nbTP(i,1)/(nbTP(i,1)+nbFN(i,1));
nbPrecision(i,1)= nbTP(i,1)/(nbTP(i,1)+nbFP(i,1));
nbFmeasure(i,1)=2/((1/nbPrecision(i,1))+(1/nbRecall(i,1)));
end
%%
e=[1:length(TP)];
figure
plot(e,sort(Accuracy),'o-g','LineWidth',1,...
'MarkerEdgeColor','g',...
'MarkerFaceColor','g',...
'MarkerSize',5)
hold on
plot(e,sort(knnAccuracy),'^-k','LineWidth',1,...
'MarkerEdgeColor','k',...
'MarkerFaceColor','k',...
'MarkerSize',5)
hold on
plot(e,sort(svmAccuracy),'+-b','LineWidth',1,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',5)
hold on
plot(e,sort(nnAccuracy),'*-r','LineWidth',1,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',5)
hold on
plot(e,sort(dtAccuracy),'x-c','LineWidth',1,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',5)
hold on
plot(e,sort(nbAccuracy),'.-m','LineWidth',1,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
xlabel('Number of instances* 10')
xlim([1 length(TP)])
ylabel('Accuracy')
% ylim([0 1])
legend('Proposed method','KNN','SVM','NN','DT','NB','location','northoutside','Orientation','horizontal')
title('Comparison of the accuracy of the proposed method and classifications')
%%
figure
plot(e,sort(1-Accuracy),'o-g','LineWidth',1,...
'MarkerEdgeColor','g',...
'MarkerFaceColor','g',...
'MarkerSize',5)
hold on
plot(e,sort(1-knnAccuracy),'^-k','LineWidth',1,...
'MarkerEdgeColor','k',...
'MarkerFaceColor','k',...
'MarkerSize',5)
hold on
plot(e,sort(1-svmAccuracy),'+-b','LineWidth',1,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',5)
hold on
plot(e,sort(1-nnAccuracy),'*-r','LineWidth',1,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',5)
hold on
plot(e,sort(1-dtAccuracy),'x-c','LineWidth',1,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',5)
hold on
plot(e,sort(1-nbAccuracy),'.-m','LineWidth',1,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
xlabel('Number of instances* 10')
xlim([1 length(TP)])
ylabel('Error')
% ylim([0 1])
legend('Proposed method','KNN','SVM','NN','DT','NB','location','northoutside','Orientation','horizontal')
title('Comparison of the error of the proposed method and classifications')
%%
figure
plot(e,sort(Recall),'o-g','LineWidth',1,...
'MarkerEdgeColor','g',...
'MarkerFaceColor','g',...
'MarkerSize',5)
hold on
plot(e,sort(knnRecall),'^-k','LineWidth',1,...
'MarkerEdgeColor','k',...
'MarkerFaceColor','k',...
'MarkerSize',5)
hold on
plot(e,sort(svmRecall),'+-b','LineWidth',1,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',5)
hold on
plot(e,sort(nnRecall),'*-r','LineWidth',1,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',5)
hold on
plot(e,sort(dtRecall),'x-c','LineWidth',1,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',5)
hold on
plot(e,sort(nbRecall),'.-m','LineWidth',1,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
xlabel('Number of instances* 10')
xlim([1 length(TP)])
ylabel('Recall')
% ylim([0 1])
legend('Proposed method','KNN','SVM','NN','DT','NB','location','northoutside','Orientation','horizontal')
title('Comparison of the Recall of the proposed method and classifications')
%%

figure
plot(e,sort(Precision),'o-g','LineWidth',1,...
'MarkerEdgeColor','g',...
'MarkerFaceColor','g',...
'MarkerSize',5)
hold on
plot(e,sort(knnPrecision),'^-k','LineWidth',1,...
'MarkerEdgeColor','k',...
'MarkerFaceColor','k',...
'MarkerSize',5)
hold on
plot(e,sort(svmPrecision),'+-b','LineWidth',1,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',5)
hold on
plot(e,sort(nnPrecision),'*-r','LineWidth',1,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',5)
hold on
plot(e,sort(dtPrecision),'x-c','LineWidth',1,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',5)
hold on
plot(e,sort(nbPrecision),'.-m','LineWidth',1,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
xlabel('Number of instances* 10')
xlim([1 length(TP)])
ylabel('Precision')
% ylim([0 1])
legend('Proposed method','KNN','SVM','NN','DT','NB','location','northoutside','Orientation','horizontal')
title('Comparison of the Precision of the proposed method and classifications')%%
%%
figure
plot(e,sort(TruePositiveRate),'o-g','LineWidth',1,...
'MarkerEdgeColor','g',...
'MarkerFaceColor','g',...
'MarkerSize',5)
hold on
plot(e,sort(knnTruePositiveRate),'^-k','LineWidth',1,...
'MarkerEdgeColor','k',...
'MarkerFaceColor','k',...
'MarkerSize',5)
hold on
plot(e,sort(svmTruePositiveRate),'+-b','LineWidth',1,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',5)
hold on
plot(e,sort(nnTruePositiveRate),'*-r','LineWidth',1,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',5)
hold on
plot(e,sort(dtTruePositiveRate),'x-c','LineWidth',1,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',5)
hold on
plot(e,sort(nbTruePositiveRate),'.-m','LineWidth',1,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
xlabel('Number of instances* 10')
xlim([1 length(TP)])
ylabel('TruePositiveRate')
% ylim([0 1])
legend('Proposed method','KNN','SVM','NN','DT','NB','location','northoutside','Orientation','horizontal')
title('Comparison of the TruePositiveRate of the proposed method and classifications')
%%
figure
plot(e,sort(TruePositiveRate),'o-g','LineWidth',1,...
'MarkerEdgeColor','g',...
'MarkerFaceColor','g',...
'MarkerSize',5)
hold on
plot(e,sort(knnTrueNegativeRate),'^-k','LineWidth',1,...
'MarkerEdgeColor','k',...
'MarkerFaceColor','k',...
'MarkerSize',5)
hold on
plot(e,sort(svmTrueNegativeRate),'+-b','LineWidth',1,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',5)
hold on
plot(e,sort(nnTrueNegativeRate),'*-r','LineWidth',1,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',5)
hold on
plot(e,sort(dtTrueNegativeRate),'x-c','LineWidth',1,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',5)
hold on
plot(e,sort(nbTrueNegativeRate),'.-m','LineWidth',1,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
xlabel('Number of instances* 10')
xlim([1 length(TP)])
ylabel('TrueNegativeRate')
% ylim([0 1])
legend('Proposed method','KNN','SVM','NN','DT','NB','location','northoutside','Orientation','horizontal')
title('Comparison of the TrueNegativeRate of the proposed method and classifications')
%%
figure
plot(e,sort(Fmeasure),'o-g','LineWidth',1,...
'MarkerEdgeColor','g',...
'MarkerFaceColor','g',...
'MarkerSize',5)
hold on
plot(e,sort(knnFmeasure),'^-k','LineWidth',1,...
'MarkerEdgeColor','k',...
'MarkerFaceColor','k',...
'MarkerSize',5)
hold on
plot(e,sort(svmFmeasure),'+-b','LineWidth',1,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',5)
hold on
plot(e,sort(nnFmeasure),'*-r','LineWidth',1,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',5)
hold on
plot(e,sort(dtFmeasure),'x-c','LineWidth',1,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',5)
hold on
plot(e,sort(nbFmeasure),'.-m','LineWidth',1,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
xlabel('Number of instances* 10')
xlim([1 length(TP)])
ylabel('F-measure')
% ylim([0 1])
legend('Proposed method','KNN','SVM','NN','DT','NB','location','northoutside','Orientation','horizontal')
title('Comparison of the F-measure of the proposed method and classifications')
%%
accuracy=mean(Accuracy)
truePositiveRate = mean(TruePositiveRate)
trueNegativeRate = mean(TruePositiveRate)
recall= mean(Recall)
precision= mean(Precision)
fmeasure=mean(Fmeasure)
%%
knnaccuracy=mean(knnAccuracy)
knntruePositiveRate = mean(knnTruePositiveRate)
knntrueNegativeRate = mean(knnTrueNegativeRate)
knnrecall= mean(knnRecall)
knnprecision= mean(knnPrecision)
knnfmeasure=mean(knnFmeasure)
%%
svmaccuracy=mean(svmAccuracy)
svmtruePositiveRate = mean(svmTruePositiveRate)
svmtrueNegativeRate = mean(svmTrueNegativeRate)
svmrecall= mean(svmRecall)
svmprecision= mean(svmPrecision)
svmfmeasure=mean(svmFmeasure)
%%
nnaccuracy= mean(nnAccuracy)
nntruePositiveRate = mean(nnTruePositiveRate)
nntrueNegativeRate = mean(nnTrueNegativeRate)
nnrecall= mean(nnRecall)
nnprecision= mean(nnPrecision)
nnfmeasure=mean(nnFmeasure)
%%
dtaccuracy=mean(dtAccuracy)
dttruePositiveRate = mean(dtTruePositiveRate)
dttrueNegativeRate = mean(dtTrueNegativeRate)
dtrecall= mean(dtRecall)
dtprecision= mean(dtPrecision)
dtfmeasure=mean(dtFmeasure)
%%
nbaccuracy=mean(nbAccuracy)
nbtruePositiveRate = mean(nbTruePositiveRate)
nbtrueNegativeRate = mean(nbTrueNegativeRate)
nbrecall=mean(nbRecall)
nbprecision= mean(nbPrecision)
nbfmeasure=mean(nbFmeasure)


