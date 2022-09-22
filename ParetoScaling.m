% Comparing conditional Asmussen--Kroese estimator,conditional Asmussen--Kroese and importance sampling estimator 
% All algorithms combined with control variates
% Including both conditioning on scaling random variable and conditioning on phase-type (PH) random variable
% Compare ConAK+CV and ConAK+IS+CV on Pareto, ConAK+CV and ConAK+IS+CV on PH 
% Take Pareto*Erlang for example
clear all;clc;
rho=0.8;                  %Parameter of geometric distribution
lambda1=3;                %Parameters of Erlang distribution
nEr = 2;
alpha_par=8; sig_par=1;   %Parameters of Pareto distribution
k_par=1/alpha_par; sigma_par=sig_par/alpha_par; theta_par=sig_par;
N1=100; N=10^5;
gammaU=linspace(1,150000,N1);%The threshold

%% The following is conditional Asmussen--Kroese algorithm conditioning on scaling random variable
ell_conW = nan(1,length(gammaU)); RE_conW = nan(1,length(gammaU));  LogEff_conW = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_conW = nan(1,N);
    for i=1:N
        R1=1;
        while rand<rho
            R1=R1+1;
        end
        if R1==1
            val1=gammaU(j);
        else
            Y1_1=gamrnd(nEr,1/lambda1,1,R1-1);
            Y1_2=gprnd(k_par,sigma_par,theta_par,1,R1-1);
            S1=sum(Y1_1.*Y1_2);MM1=max(Y1_1.*Y1_2);
            val1=max(MM1,gammaU(j)-S1);
        end                   
            Y1_conW=gprnd(k_par,sigma_par,theta_par);
            Y_conW(i)=R1*gamcdf(val1/Y1_conW,nEr,1/lambda1,'upper')-(R1-1/(1-rho)).*gamcdf(gammaU(j)/Y1_conW,nEr,1/lambda1,'upper');
    end
    ell_conW(j)=mean(Y_conW)*rho;
    RE_conW(j)=std(Y_conW*rho)/sqrt(N)/ell_conW(j);
    LogEff_conW(j)=log(var(Y_conW))/log(ell_conW(j)^2);
end

%% The following is conditional Asmussen--Kroese conditioning on scaling random variable + IS algorithm
ell_conWIS = nan(1,length(gammaU)); RE_conWIS = nan(1,length(gammaU));  LogEff_conWIS = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_conWIS = nan(1,N);
    for i=1:N
        R2=1;
        while rand<rho
            R2=R2+1;
        end
        if R2==1
            val2=gammaU(j);           
        else
            Y2_1=gamrnd(nEr,1/lambda1,1,R2-1);
            Y2_2=gprnd(k_par,sigma_par,theta_par,1,R2-1);
            S2=sum(Y2_1.*Y2_2);MM2=max(Y2_1.*Y2_2);
            val2=max(MM2,gammaU(j)-S2);           
        end
        theta=1-1./(alpha_par*log(val2*lambda1));
        alpha_par1=alpha_par*(1-theta); k_par1=1/alpha_par1; sigma_par1=sig_par/alpha_par1;
        Y2conIS=gprnd(k_par1,sigma_par1,theta_par);
        Y_conWIS(i)=(R2*gamcdf(val2/Y2conIS,nEr,1/lambda1,'upper')-(R2-1/(1-rho)).*gamcdf(gammaU(j)/Y2conIS,nEr,1/lambda1,'upper'))*(Y2conIS^(-theta*alpha_par))*(1/(1-theta));
    end
    ell_conWIS(j)=mean(Y_conWIS)*rho;
    RE_conWIS(j)=std(Y_conWIS*rho)/sqrt(N)/ell_conWIS(j);
    LogEff_conWIS(j)=log(var(Y_conWIS))/log(ell_conWIS(j)^2);
end

%% The following is Conditional Asmussen--Kroese algorithm conditioning on PH 
ell_conPH = nan(1,length(gammaU)); RE_conPH = nan(1,length(gammaU));  LogEff_conPH = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_AconPH = nan(1,N);
    for i=1:N
        R3=1;
        while rand<rho
            R3=R3+1;
        end
        if R3==1
            val3=gammaU(j);
        else
            Y3_1=gamrnd(nEr,1/lambda1,1,R3-1);
            Y3_2=gprnd(k_par,sigma_par,theta_par,1,R3-1);
            S3=sum(Y3_1.*Y3_2);MM3=max(Y3_1.*Y3_2);
            val3=max(MM3,gammaU(j)-S3);
        end
        Y_PH=gamrnd(nEr,1/lambda1);
        Y_AconPH(i)=R3*gpcdf(val3/Y_PH,k_par,sigma_par,theta_par,'upper')-(R3-1/(1-rho)).*gpcdf(gammaU(j)/Y_PH,k_par,sigma_par,theta_par,'upper');        
    end
    ell_conPH(j)=mean(Y_AconPH)*rho;
    RE_conPH(j)=std(Y_AconPH*rho)/sqrt(N)/ell_conPH(j);
    LogEff_conPH(j)=log(var(Y_AconPH))/log(ell_conPH(j)^2);
end

%% The following is Conditional Asmussen--Kroese conditioning on PH + IS algorithm (Exponential)
ell_conPHIS2 = nan(1,length(gammaU)); RE_conPHIS2 = nan(1,length(gammaU));  LogEff_conPHIS2 = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_conPHIS2 = nan(1,N);
    for i=1:N
        R5=1;
        while rand<rho
            R5=R5+1;
        end
        if R5==1
            val5=gammaU(j);           
        else
            Y5_1=gamrnd(nEr,1/lambda1,1,R5-1);
            Y5_2=gprnd(k_par,sigma_par,theta_par,1,R5-1);
            S5=sum(Y5_1.*Y5_2);MM5=max(Y5_1.*Y5_2);
            val5=max(MM5,gammaU(j)-S5);            
        end
        lambdaISExp = 1/(val5*(alpha_par-1)/alpha_par);
        Y2_conPHIS=exprnd(1/lambdaISExp);
        L5 = (lambda1/(lambda1-lambdaISExp))^nEr/lambdaISExp*gampdf(Y2_conPHIS,nEr,1/(lambda1-lambdaISExp));
        Y_conPHIS2(i)=(R5*gpcdf(val5/Y2_conPHIS,k_par,sigma_par,theta_par,'upper')-(R5-1/(1-rho)).*gpcdf(gammaU(j)/Y2_conPHIS,k_par,sigma_par,theta_par,'upper'))*L5;       
    end
    ell_conPHIS2(j)=mean(Y_conPHIS2)*rho;
    RE_conPHIS2(j)=std(Y_conPHIS2*rho)/sqrt(N)/ell_conPHIS2(j);
    LogEff_conPHIS2(j)=log(var(Y_conPHIS2))/log(ell_conPHIS2(j)^2);
end

%% The following is Conditional Asmussen--Kroese conditioning on PH + IS algorithm (Erlang)
ell_conPHIS = nan(1,length(gammaU)); RE_conPHIS = nan(1,length(gammaU));  LogEff_conPHIS = nan(1,length(gammaU));
for j=1:length(gammaU)
    Y_conPHIS = nan(1,N);
    for i=1:N
        R4=1;
        while rand<rho
            R4=R4+1;
        end
        if R4==1
            val4=gammaU(j);           
        else
            Y4_1=gamrnd(nEr,1/lambda1,1,R4-1);
            Y4_2=gprnd(k_par,sigma_par,theta_par,1,R4-1);
            S4=sum(Y4_1.*Y4_2);MM4=max(Y4_1.*Y4_2);
            val4=max(MM4,gammaU(j)-S4);
        end
        lambdaIS = 1/(val4*(alpha_par-1)/nEr/alpha_par);
        Y2_conPHIS=gamrnd(nEr,1/lambdaIS);       
        L4 = (lambda1/lambdaIS)^nEr*exp(-(lambda1-lambdaIS).*Y2_conPHIS);
        Y_conPHIS(i)=(R4*gpcdf(val4/Y2_conPHIS,k_par,sigma_par,theta_par,'upper')-(R4-1/(1-rho)).*gpcdf(gammaU(j)/Y2_conPHIS,k_par,sigma_par,theta_par,'upper'))*L4;
    end
    ell_conPHIS(j)=mean(Y_conPHIS)*rho;
    RE_conPHIS(j)=std(Y_conPHIS*rho)/sqrt(N)/ell_conPHIS(j);
    LogEff_conPHIS(j)=log(var(Y_conPHIS))/log(ell_conPHIS(j)^2);
end

figure(1)
% semilogy(gammaU,ell_conW,'go')
% hold on
semilogy(gammaU,ell_conWIS,'ks')
hold on
semilogy(gammaU,ell_conPH,'r*')
hold on
semilogy(gammaU,ell_conPHIS2,'b.')
hold on
semilogy(gammaU,ell_conPHIS,'cd')
hold off
xlabel('u')
set(gca,'fontsize',12.25)

figure(2)
% plot(gammaU,RE_conW,'go')
% hold on
plot(gammaU,RE_conWIS,'ks')
hold on
plot(gammaU,RE_conPH,'r*')
hold on
plot(gammaU,RE_conPHIS2,'b.')
hold on
plot(gammaU,RE_conPHIS,'cd')
hold off
xlabel('u','FontSize',12)
set(gca,'fontsize',12.25)

figure(3)
% plot(gammaU,LogEff_conW,'go')
% hold on
plot(gammaU,LogEff_conWIS,'ks')
hold on
plot(gammaU,LogEff_conPH,'r*')
hold on
plot(gammaU,LogEff_conPHIS2,'b.')
hold on
plot(gammaU,LogEff_conPHIS,'cd')
hold off
xlabel('u','FontSize',12)
ylim([0 1])
set(gca,'fontsize',12.25)