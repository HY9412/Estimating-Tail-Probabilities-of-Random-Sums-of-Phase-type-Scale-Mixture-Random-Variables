% Comparing conditional Asmussen--Kroese estimator,conditional Asmussen--Kroese and importance sampling estimator 
% All algorithms combined with control variates
% Including both conditioning on scaling random variable and conditioning on phase-type (PH) random variable
% Number of summands is deterministic
% Take Weibull*Erlang for example
clear all;clc;
n1 = 15;                                            % number of summands
nEr = 2;    lambda1 = 3;                            % parameters of Erlang distribution
a = 2;      b = 0.35;                               % parameters of scaling distribution
N1 = 100;   N = 10^5;
gammaU = linspace(1,150000,N1);                     % The thresholds

%% The following is ConAK algorithm conditional on Weibull
ell_onW = nan(1,length(gammaU)); RE_onW = nan(1,length(gammaU)); LE_onW = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_AtsComIS = nan(1,N);
    for i=1:N
        Y_1{i}=gamrnd(nEr,1/lambda1,1,n1-1);
        Y_2{i}=wblrnd(a,b,1,n1-1);
        S(i)=sum(Y_1{i}.*Y_2{i});MM(i)=max(Y_1{i}.*Y_2{i});
        val=max(MM(i),gammaU(j)-S(i));
        Wb1=wblrnd(a,b);
        Y_AtsComIS(i)=n1*gamcdf(val/Wb1,nEr,1/lambda1,'upper');
    end
    ell_onW(j) = mean(Y_AtsComIS);
    RE_onW(j) = std(Y_AtsComIS)/sqrt(N)/ell_onW(j);
    LE_onW(j) = log(var(Y_AtsComIS))/log(ell_onW(j)^2);
end

%% The following is ConAK algorithm conditional on Weibull + IS
ell_onWIS = nan(1,length(gammaU)); RE_onWIS = nan(1,length(gammaU)); LE_onWIS = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_AtsComIS2 = nan(1,N);
    for i=1:N
        Y_1IS{i}=gamrnd(nEr,1/lambda1,1,n1-1);
        Y_2IS{i}=wblrnd(a,b,1,n1-1);
        S(i)=sum(Y_1IS{i}.*Y_2IS{i});MM(i)=max(Y_1IS{i}.*Y_2IS{i});
        valIS=max(MM(i),gammaU(j)-S(i));
        theta1=1+1./log((wblcdf(valIS*lambda1/nEr,a,b,'upper')));
        aa=1-rand^(1/(1-theta1));
        S32(i)=wblinv(aa,a,b);                               %Sample from the hazard rate twisting S
        Y_AtsComIS2(i)=(n1*gamcdf(valIS/S32(i),nEr,1/lambda1,'upper'))*exp(-theta1*(S32(i)/a)^b)*(1/(1-theta1));
    end
    ell_onWIS(j) = mean(Y_AtsComIS2);
    RE_onWIS(j) = std(Y_AtsComIS2)/sqrt(N)/ell_onWIS(j);
    LE_onWIS(j) = log(var(Y_AtsComIS2))/log(ell_onWIS(j)^2);
end

%% The following is ConAK algorithm conditional on Erlang
ell_onPH = nan(1,length(gammaU)); RE_onPH = nan(1,length(gammaU)); LE_onPH = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_Ats = nan(1,N);
    for i=1:N
        Y_11=gamrnd(nEr,1/lambda1,1,n1-1);
        Y_22=wblrnd(a,b,1,n1-1);
        SS=sum(Y_11.*Y_22);MMM=max(Y_11.*Y_22);
        val1=max(MMM,gammaU(j)-SS);
        Yts=gamrnd(nEr,1/lambda1);
        Y_Ats(i)=n1*wblcdf(val1/Yts,a,b,'upper');
    end
    ell_onPH(j) = mean(Y_Ats);
    RE_onPH(j) = std(Y_Ats)/sqrt(N)/ell_onPH(j);
    LE_onPH(j)= log(var(Y_Ats))/log(ell_onPH(j)^2);
end

%% The following is ConAK algorithm conditional on Erlang + IS (density selection Exponential)
ell_onPHIS = nan(1,length(gammaU)); RE_onPHIS = nan(1,length(gammaU)); LE_onPHIS = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_Ats2 = nan(1,N);
    for i=1:N
        Y_11IS=gamrnd(nEr,1/lambda1,1,n1-1);
        Y_22IS=wblrnd(a,b,1,n1-1);
        SSIS=sum(Y_11IS.*Y_22IS);MMM=max(Y_11IS.*Y_22IS);
        val1IS=max(MMM,gammaU(j)-SSIS);
        lambdaIS = 1*(a*gamma(1+1/b))/val1IS;
        YtsIS=exprnd(1/lambdaIS);
        L = (1/lambdaIS)*(lambda1/(lambda1-lambdaIS))^nEr*gampdf(YtsIS,nEr,1/(lambda1-lambdaIS));
        Y_Ats2(i)=(n1*wblcdf(val1IS/YtsIS,a,b,'upper'))*L;
    end
    ell_onPHIS(j) = mean(Y_Ats2);
    RE_onPHIS(j) = std(Y_Ats2)/sqrt(N)/ell_onPHIS(j);
    LE_onPHIS(j)=abs(log(var(Y_Ats2))/log(ell_onPHIS(j)^2));
end

%% The following is ConAK algorithm conditional on Erlang + IS (density selection Erlang)
ell_onPHIS2 = nan(1,length(gammaU)); RE_onPHIS2 = nan(1,length(gammaU)); LE_onPHIS2 = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_Ats2IS = nan(1,N);
    for i=1:N
        Y_11ISER=gamrnd(nEr,1/lambda1,1,n1-1);
        Y_22ISER=wblrnd(a,b,1,n1-1);
        SSISER=sum(Y_11ISER.*Y_22ISER);MMM=max(Y_11ISER.*Y_22ISER);
        val1ISER=max(MMM,gammaU(j)-SSISER);
        lambdaIS2 = nEr*(1)*(a*gamma(1+1/b))/val1ISER;
        YtsISER=gamrnd(nEr,1/lambdaIS2);
        L2 = (lambda1/lambdaIS2)^nEr*exp(-(lambda1-lambdaIS2)*YtsISER);
        Y_Ats2IS(i)=(n1*wblcdf(val1ISER/YtsISER,a,b,'upper'))*L2;
    end
    ell_onPHIS2(j) = mean(Y_Ats2IS);
    RE_onPHIS2(j) = std(Y_Ats2IS)/sqrt(N)/ell_onPHIS2(j);
    LE_onPHIS2(j)= log(var(Y_Ats2IS))/log(ell_onPHIS2(j)^2);
end
figure(1)
% semilogy(gammaU,ell_onW,'go')
% hold on
semilogy(gammaU,ell_onWIS,'ks')
hold on
semilogy(gammaU,ell_onPH,'r*')
hold on
semilogy(gammaU,ell_onPHIS,'b.')
hold on
semilogy(gammaU,ell_onPHIS2,'cd')
hold off
xlabel('u')
set(gca,'fontsize',12.25)

figure(2)
% plot(gammaU,RE_onW,'go')
% hold on
plot(gammaU,RE_onWIS,'ks')
hold on
plot(gammaU,RE_onPH,'r*')
hold on
plot(gammaU,RE_onPHIS,'b.')
hold on
plot(gammaU,RE_onPHIS2,'cd')
hold off
xlabel('u','FontSize',12)
% ylim([0 1.5])
set(gca,'fontsize',12.25)

figure(3)
% plot(gammaU,LE_onW,'go')
% hold on
plot(gammaU,LE_onWIS,'ks')
hold on
plot(gammaU,LE_onPH,'r*')
hold on
plot(gammaU,LE_onPHIS,'b.')
hold on
plot(gammaU,LE_onPHIS2,'cd')
hold off
xlabel('u','FontSize',12)
% ylim([0 1])
set(gca,'fontsize',12.25)

% save WeibulN30.mat

