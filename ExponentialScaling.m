% Comparing conditional Asmussen--Kroese estimator,conditional Asmussen--Kroese and importance sampling estimator 
% All algorithms combined with control variates
% Including both conditioning on scaling random variable and conditioning on phase-type (PH) random variable
% Compare ConAK1+CV, ConAK1+IS+CV, and ConAK2+IS+CV on PH (includes using
% an exponential PDF and an Erlang PDF)
% Take Exponential*Erlang for example
clear all;clc;
rho=0.8;                          % parameter of geometric distribution
lambda2=1;                        % parameter of scaling distribution (exponential)
nEr = 1;   lambda1=3;             % parameter of Erlang distribution 
N1=100; N=10^5;                   
gammaU=linspace(1,10000,N1);      % The thresholds

%% The following is ConAK algorithm conditional on W (Exponential Scaling)
ell_A1 = nan(1,N1); RE_A1 = nan(1,N1); LE_A1 = nan(1,N1); 
for j=1:length(gammaU)
    Y_Ats = nan(1,N);
    for i=1:N
        R1=1;
        while rand<rho
            R1=R1+1;
        end
        if R1==1
            val=gammaU(j);
        else
            Y_1=gamrnd(nEr,1/lambda1,1,R1-1);
            Y_2=exprnd(1/lambda2,1,R1-1);
            S=sum(Y_1.*Y_2);MM=max(Y_1.*Y_2);
            val=max(MM,gammaU(j)-S);
        end      
        Yts=exprnd(1/lambda2);
        Y_Ats(i)=R1*gamcdf(val/Yts,nEr,1/lambda1,'upper')-(R1-1/(1-rho)).*gamcdf(gammaU(j)/Yts,nEr,1/lambda1,'upper');
    end
    ell_A1(j)=mean(Y_Ats)*rho;
    RE_A1(j)=std(Y_Ats)/sqrt(N)/ell_A1(j);
    LE_A1(j)=log(var(Y_Ats))/log(ell_A1(j)^2);
end

%% The following is ConAK algorithm conditional + IS on W (Exponential Scaling)
ell_A1conIS = nan(1,N1); RE_A1conIS = nan(1,N1);LE_A1conIS = nan(1,N1);
for j=1:length(gammaU)
    Y_AtsconIS = nan(1,N);
    for i=1:N        
        R2=1;
        while rand<rho
            R2=R2+1;
        end
        if R2==1
            valconIS=gammaU(j);
        else
            Y_1conIS=gamrnd(nEr,1/lambda1,1,R2-1);
            Y_2conIS=exprnd(1/lambda2,1,R2-1);
            S_conIS=sum(Y_1conIS.*Y_2conIS);MM_conIS=max(Y_1conIS.*Y_2conIS);
            valconIS=max(MM_conIS,gammaU(j)-S_conIS);
        end
        lambda3 = nEr./(valconIS*lambda1);
        Yts_conIS = exprnd(1/lambda3);
        L_conIS = (lambda2/lambda3).*exp(-(lambda2-lambda3).*Yts_conIS);
        Y_AtsconIS(i)=(R2*gamcdf(valconIS/Yts_conIS,nEr,1/lambda1,'upper')-(R2-1/(1-rho)).*gamcdf(gammaU(j)/Yts_conIS,nEr,1/lambda1,'upper'))*L_conIS;
    end
    ell_A1conIS(j)=mean(Y_AtsconIS)*rho;
    RE_A1conIS(j)=std(Y_AtsconIS)/sqrt(N)/ell_A1conIS(j);
    LE_A1conIS(j)=log(var(Y_AtsconIS))/log(ell_A1conIS(j)^2);
end

%% The following is ConAK algorithm conditional on Erlang (PH)
ell_A1PH = nan(1,N1); RE_A1PH = nan(1,N1); LE_A1PH = nan(1,N1); 
for j=1:length(gammaU)
    Y_AtsPH = nan(1,N); 
    for i=1:N        
        R3=1;
        while rand<rho
            R3=R3+1;
        end
        if R3==1
            valPH=gammaU(j);
        else
            Y_1PH=gamrnd(nEr,1/lambda1,1,R3-1);
            Y_2PH=exprnd(1/lambda2,1,R3-1);
            S_PH=sum(Y_1PH.*Y_2PH);MM_PH=max(Y_1PH.*Y_2PH);
            valPH=max(MM_PH,gammaU(j)-S_PH);
        end
        YtsPH=gamrnd(nEr,1/lambda1);
        Y_AtsPH(i)=R3*expcdf(valPH/YtsPH,1/lambda2,'upper')-(R3-1/(1-rho)).*expcdf(gammaU(j)/YtsPH,1/lambda2,'upper');
    end 
    % Conditioning on Erlang
    ell_A1PH(j)=mean(Y_AtsPH)*rho;
    RE_A1PH(j)=std(Y_AtsPH)/sqrt(N)/ell_A1PH(j);
    LE_A1PH(j)=log(var(Y_AtsPH))/log(ell_A1PH(j)^2);
end

%% The following is ConAK algorithm + IS conditional on Erlang (PH) Exponential PDF
ell_A1PHIS2 = nan(1,N1); RE_A1PHIS2 = nan(1,N1); LE_A1PHIS2 = nan(1,N1); 
for j=1:length(gammaU)
    Y_AtsPHIS2 = nan(1,N); 
    for i=1:N        
        R5=1;
        while rand<rho
            R5=R5+1;
        end
        if R5==1
            val_PHIS2=gammaU(j);
        else
            Y_1PHIS2=gamrnd(nEr,1/lambda1,1,R5-1);
            Y_2PHIS2=exprnd(1/lambda2,1,R5-1);
            S_PHIS2=sum(Y_1PHIS2.*Y_2PHIS2);MM_PHIS2=max(Y_1PHIS2.*Y_2PHIS2);
            val_PHIS2=max(MM_PHIS2,gammaU(j)-S_PHIS2);
        end
        lambda3PHExp=1/(val_PHIS2*lambda2);
        YtsPH2=exprnd(1/lambda3PHExp);
        L_PHIS2 = (lambda1^nEr/lambda3PHExp).*YtsPH2^(nEr-1)*exp(-(lambda1-lambda3PHExp).*YtsPH2)/factorial(nEr-1);
        Y_AtsPHIS2(i)=(R5*expcdf(val_PHIS2/YtsPH2,1/lambda2,'upper')-(R5-1/(1-rho)).*expcdf(gammaU(j)/YtsPH2,1/lambda2,'upper'))*L_PHIS2;
    end 
    % Conditioning on Erlang
    ell_A1PHIS2(j)=mean(Y_AtsPHIS2)*rho;
    RE_A1PHIS2(j)=std(Y_AtsPHIS2)/sqrt(N)/ell_A1PHIS2(j);
    LE_A1PHIS2(j)=log(var(Y_AtsPHIS2))/log(ell_A1PHIS2(j)^2);
end

%% The following is ConAK algorithm + IS conditional on Erlang (PH) using an Erlang PDF
ell_A1PHIS = nan(1,N1); RE_A1PHIS = nan(1,N1); LE_A1PHIS = nan(1,N1); 
for j=1:length(gammaU)
    Y_AtsPHIS = nan(1,N); 
    for i=1:N        
        R4=1;
        while rand<rho
            R4=R4+1;
        end
        if R4==1
            val_PHIS=gammaU(j);
        else
            Y_1PHIS=gamrnd(nEr,1/lambda1,1,R4-1);
            Y_2PHIS=exprnd(1/lambda2,1,R4-1);
            S_PHIS=sum(Y_1PHIS.*Y_2PHIS);MM_PHIS=max(Y_1PHIS.*Y_2PHIS);
            val_PHIS=max(MM_PHIS,gammaU(j)-S_PHIS);
        end
        lambda3PH=nEr./(val_PHIS*lambda2);
        YtsPH=gamrnd(nEr,1/lambda3PH);
        L_PHIS = (lambda1/lambda3PH)^nEr.*exp(-(lambda1-lambda3PH).*YtsPH);
        Y_AtsPHIS(i)=(R4*expcdf(val_PHIS/YtsPH,1/lambda2,'upper')-(R4-1/(1-rho)).*expcdf(gammaU(j)/YtsPH,1/lambda2,'upper'))*L_PHIS;
    end 
    % Conditioning on Erlang
    ell_A1PHIS(j)=mean(Y_AtsPHIS)*rho;
    RE_A1PHIS(j)=std(Y_AtsPHIS)/sqrt(N)/ell_A1PHIS(j);
    LE_A1PHIS(j)=log(var(Y_AtsPHIS))/log(ell_A1PHIS(j)^2);
end
figure(1)
% semilogy(gammaU,ell_A1,'go') 
% hold on
semilogy(gammaU,ell_A1conIS,'ks') 
hold on
semilogy(gammaU,ell_A1PH,'r*') 
hold on
semilogy(gammaU,ell_A1PHIS2,'b.') 
hold on
semilogy(gammaU,ell_A1PHIS,'cd') 
hold off
xlabel('u')
set(gca,'fontsize',12.25)

figure(2)
% plot(gammaU,RE_A1,'go')
% hold on
plot(gammaU,RE_A1conIS,'ks')
hold on
plot(gammaU,RE_A1PH,'r*')
hold on
plot(gammaU,RE_A1PHIS2,'b.')
hold on
plot(gammaU,RE_A1PHIS,'cd')
hold off
xlabel('u','FontSize',12)
% ylim([0 0.05])
set(gca,'fontsize',12.25)

figure(3)
% plot(gammaU,LE_A1,'go')
% hold on
plot(gammaU,LE_A1conIS,'ks')
hold on
plot(gammaU,LE_A1PH,'r*')
hold on
plot(gammaU,LE_A1PHIS2,'b.')
hold on
plot(gammaU,LE_A1PHIS,'cd')
hold off
xlabel('u','FontSize',12)
% ylim([0 1])
set(gca,'fontsize',12.25)