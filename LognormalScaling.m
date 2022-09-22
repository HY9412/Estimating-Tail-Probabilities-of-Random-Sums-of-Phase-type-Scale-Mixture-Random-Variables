% Comparing conditional Asmussen--Kroese estimator,conditional Asmussen--Kroese and importance sampling estimator 
% All algorithms combined with control variates
% Including both conditioning on scaling random variable and conditioning on phase-type (PH) random variable
% Compare ConAK+CV and ConAK+IS+CV on lognormal, ConAK+CV and ConAK+IS+CV on PH 
% Take Lognormal*Erlang for example
clear all;clc;
rho=0.8;                                  %parameter of geometric distribution
nEr = 2;   lambda1=3;                     % parameter of Erlang distribution
mu = 2; sigma = 1.5;                      % parameter of scaling distribution (lognormal)
meanLN = exp(mu+sigma^2/2);
N1=100; N=10^5;
gammaT=linspace(1,150000,N1);%The threshold
%% The following is conditional AK  algorithm on W
ell_conW = nan(1,length(gammaT)); RE_conW = nan(1,length(gammaT)); LE_conW = nan(1,length(gammaT));
for j=1:length(gammaT)
    Y_A = nan(1,N);
    for i=1:N
        R1=1;
        while rand<rho
            R1=R1+1;
        end
        if R1==1
            val=gammaT(j);
        else
            Y_1=gamrnd(nEr,1/lambda1,1,R1-1);
            Y_2=lognrnd(mu,sigma,1,R1-1);
            S=sum(Y_1.*Y_2);MM=max(Y_1.*Y_2);
            val=max(MM,gammaT(j)-S);                   
        end     
        Y2 = lognrnd(mu,sigma);  
        Y_A(i)=R1*gamcdf(val/Y2,nEr,1/lambda1,'upper')-(R1-1/(1-rho)).*gamcdf(gammaT(j)/Y2,nEr,1/lambda1,'upper');
    end
    ell_conW(j)=mean(Y_A)*rho;
    RE_conW(j)=std(Y_A)/sqrt(N)/ell_conW(j);
    LE_conW(j)=log(var(Y_A))/log(ell_conW(j)^2);
 end

%% The following is conditional AK + IS algorithm on W
ell_conWIS = nan(1,length(gammaT)); RE_conWIS = nan(1,length(gammaT)); LE_conWIS = nan(1,length(gammaT)); 
for j=1:length(gammaT)
    Y_conWIS = nan(1,N);
    for i=1:N
        R2=1;
        while rand<rho
            R2=R2+1;
        end
        if R2==1
            val_conWIS=gammaT(j);
        else
            Y_conWIS1=gamrnd(nEr,1/lambda1,1,R2-1);
            Y_conWIS2=lognrnd(mu,sigma,1,R2-1);
            S_conWIS=sum(Y_conWIS1.*Y_conWIS2);MM_conWIS=max(Y_conWIS1.*Y_conWIS2);
            val_conWIS=max(MM_conWIS,gammaT(j)-S_conWIS);                   
        end     
        theta1=1+1/log(logncdf(val_conWIS*lambda1/nEr,mu,sigma,'upper'));
        a=1-(rand^(1/(1-theta1)));
        S32=logninv(a,mu,sigma); %Sample from the hazard rate twisting S
        S22=1-logncdf(S32,mu,sigma);
        L = S22^theta1*(1/(1-theta1));
        Y_conWIS(i)=(R2*gamcdf(val_conWIS/S32,nEr,1/lambda1,'upper')-(R2-1/(1-rho)).*gamcdf(gammaT(j)/S32,nEr,1/lambda1,'upper'))*L;
    end
    ell_conWIS(j)=mean(Y_conWIS)*rho;
    RE_conWIS(j)=std(Y_conWIS)/sqrt(N)/ell_conWIS(j);
    LE_conWIS(j)=log(var(Y_conWIS))/log(ell_conWIS(j)^2);
end

%% The following is conditional AK  algorithm on PH
ell_conPH = nan(1,length(gammaT)); RE_conPH = nan(1,length(gammaT)); LE_conPH = nan(1,length(gammaT));
for j=1:length(gammaT)
    Y_conPH = nan(1,N);
    for i=1:N
        R3=1;
        while rand<rho
            R3=R3+1;
        end
        if R3==1
            val_conPH=gammaT(j);
        else
            Y_conPH1=gamrnd(nEr,1/lambda1,1,R3-1);
            Y_conPH2=lognrnd(mu,sigma,1,R3-1);
            S_conPH=sum(Y_conPH1.*Y_conPH2);MM_conPH=max(Y_conPH1.*Y_conPH2);
            val_conPH=max(MM_conPH,gammaT(j)-S_conPH);                   
        end     
        Y2_conPH = gamrnd(nEr,1/lambda1);  
        Y_conPH(i)=R3*logncdf(val_conPH/Y2_conPH,mu,sigma,'upper')-(R3-1/(1-rho)).*logncdf(gammaT(j)/Y2_conPH,mu,sigma,'upper');
    end
    ell_conPH(j)=mean(Y_conPH)*rho;
    RE_conPH(j)=std(Y_conPH)/sqrt(N)/ell_conPH(j);
    LE_conPH(j)=log(var(Y_conPH))/log(ell_conPH(j)^2);
 end
 
%% The following is conditional AK + IS algorithm on PH (Exponential density)
ell_conPHISExp = nan(1,length(gammaT)); RE_conPHISExp = nan(1,length(gammaT)); LE_conPHISExp = nan(1,length(gammaT)); 
for j=1:length(gammaT)
    Y_conPHISExp = nan(1,N); L4 = nan(1,N);
    for i=1:N
        R4=1;
        while rand<rho
            R4=R4+1;
        end
        if R4==1
            val_conPHISExp=gammaT(j);
        else
            Y_conPHISExp1=gamrnd(nEr,1/lambda1,1,R4-1);
            Y_conPHISExp2=lognrnd(mu,sigma,1,R4-1);
            S_conPHISExp=sum(Y_conPHISExp1.*Y_conPHISExp2);MM_conPHISExp=max(Y_conPHISExp1.*Y_conPHISExp2);
            val_conPHISExp=max(MM_conPHISExp,gammaT(j)-S_conPHISExp);                   
        end     
        lambdaISExp = meanLN/val_conPHISExp;
        Y2_conPHISExp=exprnd(1/lambdaISExp);
        L4(i) = (lambda1^nEr/lambdaISExp)*(Y2_conPHISExp^(nEr-1))*exp(-(lambda1-lambdaISExp)*Y2_conPHISExp)/factorial(nEr-1);
        Y_conPHISExp(i) = (R4*logncdf(val_conPHISExp/Y2_conPHISExp,mu,sigma,'upper')-(R4-1/(1-rho)).*logncdf(gammaT(j)/Y2_conPHISExp,mu,sigma,'upper'))*L4(i);
    end
    ell_conPHISExp(j)=mean(Y_conPHISExp)*rho;
    RE_conPHISExp(j)=std(Y_conPHISExp)/sqrt(N)/ell_conPHISExp(j);
    LE_conPHISExp(j)=log(var(Y_conPHISExp))/log(ell_conPHISExp(j)^2);
end

 %% The following is conditional AK + IS algorithm on PH (Erlang density)
 ell_conPHISEr = nan(1,length(gammaT)); RE_conPHISEr = nan(1,length(gammaT)); LE_conPHISEr = nan(1,length(gammaT)); 
for j=1:length(gammaT)
    Y_conPHISEr = nan(1,N);
    for i=1:N
        R5=1;
        while rand<rho
            R5=R5+1;
        end
        if R5==1
            val_conPHISEr=gammaT(j);
        else
            Y_conPHISEr1=gamrnd(nEr,1/lambda1,1,R5-1);
            Y_conPHISEr2=lognrnd(mu,sigma,1,R5-1);
            S_conPHISEr=sum(Y_conPHISEr1.*Y_conPHISEr2);MM_conPHIS=max(Y_conPHISEr1.*Y_conPHISEr2);
            val_conPHISEr=max(MM_conPHIS,gammaT(j)-S_conPHISEr);                   
        end     
        lambdaISEr = nEr*meanLN/val_conPHISEr;
        Y2_conPHISEr=gamrnd(nEr,1/lambdaISEr);
        L5 = (lambda1/lambdaISEr)^nEr*exp(-(lambda1-lambdaISEr)*Y2_conPHISEr);
        Y_conPHISEr(i)=(R5*logncdf(val_conPHISEr/Y2_conPHISEr,mu,sigma,'upper')-(R5-1/(1-rho)).*logncdf(gammaT(j)/Y2_conPHISEr,mu,sigma,'upper'))*L5;
    end
    ell_conPHISEr(j)=mean(Y_conPHISEr)*rho;
    RE_conPHISEr(j)=std(Y_conPHISEr)/sqrt(N)/ell_conPHISEr(j);
    LE_conPHISEr(j)=log(var(Y_conPHISEr))/log(ell_conPHISEr(j)^2);
end
figure(1)
% semilogy(gammaT,ell_conW,'go')
% hold on
semilogy(gammaT,ell_conWIS,'ks')
hold on
semilogy(gammaT,ell_conPH,'r*')
hold on
semilogy(gammaT,ell_conPHISExp,'b.')
hold on
semilogy(gammaT,ell_conPHISEr,'cd')
hold off
xlabel('u')
set(gca,'fontsize',12.25)

figure(2)
% plot(gammaT,RE_conW,'go')
% hold on
plot(gammaT,RE_conWIS,'ks')
hold on
plot(gammaT,RE_conPH,'r*')
hold on
plot(gammaT,RE_conPHISExp,'b.')
hold on
plot(gammaT,RE_conPHISEr,'cd')
hold off
xlabel('u','FontSize',12)
set(gca,'fontsize',12.25)
% ylim([0 1.5])

figure(3)
% plot(gammaT,LE_conW,'go')
% hold on
plot(gammaT,LE_conWIS,'ks')
hold on
plot(gammaT,LE_conPH,'r*')
hold on
plot(gammaT,LE_conPHISExp,'b.')
hold on
plot(gammaT,LE_conPHISEr,'cd')
hold off
xlabel('u','FontSize',12)
ylim([0 1])
set(gca,'fontsize',12.25)
