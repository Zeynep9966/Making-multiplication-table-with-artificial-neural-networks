function [net ye yv MAPE R2] = untitled(input,output,training_rate,n1,n2,lrate)
%UNTÝTLED Summary of this function goes here
%   Detailed explanation goes here
noofdata=size(input,1);
ntd=round(noofdata*training_rate);
xt=input(1:ntd,:);
xv=input(ntd+1:end,:);

yt=output(1:ntd);
yv=output(ntd+1:end);

xt=xt';
xv=xv';
yt=yt';
yv=yv';

xtn=mapminmax(xt);
xvn=mapminmax(xv);



[ytn,ps]=mapminmax(yt);
%ytv=mapminmax(yv);

net=newff(xtn,ytn,[n1,n2],{},'trainlm');
net.trainParam.lr=lrate;
net.trainParam.epochs=10000;
net.trainParam.goal=1e-1000000;
net.trainParam.show=NaN;


net=train(net,xtn,ytn);

yen=sim(net,xvn);

ye=mapminmax('reverse',yen,ps);

ye=ye';
yv=yv';

MAPE=mean((abs(ye-yv))./yv);
SStotal=sum((yv-(mean(yv))).^2);
SSeror=sum((ye-yv).^2);
R2=1-SSeror/SStotal;

end

