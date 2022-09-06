% Preliminary comparison for the estimators ConAK1,ConAK1+CV, ConAK1+IS and ConAK1+IS+CV for the scaling random variable being exponential 
clear all;clc;
rho=0.8;                             % parameter of geometric distribution
lambda2=1;                           % parameter of exponential distribution
nEr = 1;       lambda1=3;            % parameter of scaling distribution
N1 = 100; N = 10^5;
gammaU = linspace(1,10000,N1);         % The thresholds

%% The following is ConAK algorithm conditional on W
ell_A1PH = nan(1,N1); RE_A1PH = nan(1,N1); LE_A1PH = nan(1,N1); 
for j=1:length(gammaU)
    Y_AtsPH = nan(1,N); 
    for i = 1:N        
        R3 = 1;
        while rand<rho
            R3 = R3+1;
        end
        if R3 == 1
            valPH = gammaU(j);
        else
            Y_1PH = gamrnd(nEr,1/lambda1,1,R3-1);
            Y_2PH = exprnd(1/lambda2,1,R3-1);
            S_PH = sum(Y_1PH.*Y_2PH);MM_PH=max(Y_1PH.*Y_2PH);
            valPH = max(MM_PH,gammaU(j)-S_PH);
        end
        YtsPH = exprnd(1/lambda2);
        Y_AtsPH(i) = R3*gamcdf(valPH/YtsPH,nEr,1/lambda1,'upper');
    end 
    % Conditioning on Erlang
    ell_A1PH(j) = mean(Y_AtsPH)*rho;
    RE_A1PH(j) = std(Y_AtsPH)/sqrt(N)/ell_A1PH(j);
    LE_A1PH(j) = log(var(Y_AtsPH))/log(ell_A1PH(j)^2);
end

%% The following is ConAK algorithm conditional on W with Control Variates
ell_A1PHcv = nan(1,N1); RE_A1PHcv = nan(1,N1); LE_A1PHcv = nan(1,N1);
for j = 1:length(gammaU)
    Y_AtsPHcv = nan(1,N); 
    for i = 1:N        
        R3 = 1;
        while rand<rho
            R3 = R3+1;
        end
        if R3 == 1
            valPHcv = gammaU(j);
        else
            Y_1PHcv = gamrnd(nEr,1/lambda1,1,R3-1);
            Y_2PHcv = exprnd(1/lambda2,1,R3-1);
            S_PHcv = sum(Y_1PHcv.*Y_2PHcv);MM_PHcv=max(Y_1PHcv.*Y_2PHcv);
            valPHcv = max(MM_PHcv,gammaU(j)-S_PHcv);
        end
        YtsPHcv = exprnd(1/lambda2);
        Y_AtsPHcv(i) = R3*gamcdf(valPHcv/YtsPHcv,nEr,1/lambda1,'upper')-(R3-1/(1-rho)).*gamcdf(gammaU(j)/YtsPHcv,nEr,1/lambda1,'upper');
    end 
    % Conditioning on Erlang
    ell_A1PHcv(j) = mean(Y_AtsPHcv)*rho;
    RE_A1PHcv(j) = std(Y_AtsPHcv)/sqrt(N)/ell_A1PHcv(j);
    LE_A1PHcv(j) = log(var(Y_AtsPHcv))/log(ell_A1PHcv(j)^2);
end


%% The following is ConAK algorithm + IS conditional on W
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
        lambda3PHExp=nEr./(val_PHIS2*lambda1);
        YtsPH2=exprnd(1/lambda3PHExp);
        L_PHIS2 = (lambda2/lambda3PHExp).*exp(-(lambda2-lambda3PHExp).*YtsPH2);
        Y_AtsPHIS2(i)=(R5*gamcdf(val_PHIS2/YtsPH2,nEr,1/lambda1,'upper'))*L_PHIS2;
    end 
    % Conditioning on Erlang
    ell_A1PHIS2(j)=mean(Y_AtsPHIS2)*rho;
    RE_A1PHIS2(j)=std(Y_AtsPHIS2)/sqrt(N)/ell_A1PHIS2(j);
    LE_A1PHIS2(j)=log(var(Y_AtsPHIS2))/log(ell_A1PHIS2(j)^2);
end

%% The following is ConAK algorithm + IS conditional on W with Control Variates
ell_A1PHIS2cv = nan(1,N1); RE_A1PHIS2cv = nan(1,N1); LE_A1PHIS2cv = nan(1,N1);
for j=1:length(gammaU)
    Y_AtsPHIS2cv = nan(1,N); 
    for i=1:N        
        R5=1;
        while rand<rho
            R5=R5+1;
        end
        if R5==1
            val_PHIS2cv=gammaU(j);
        else
            Y_1PHIS2cv=gamrnd(nEr,1/lambda1,1,R5-1);
            Y_2PHIS2cv=exprnd(1/lambda2,1,R5-1);
            S_PHIS2cv=sum(Y_1PHIS2cv.*Y_2PHIS2cv);MM_PHIS2cv=max(Y_1PHIS2cv.*Y_2PHIS2cv);
            val_PHIS2cv=max(MM_PHIS2cv,gammaU(j)-S_PHIS2cv);
        end
        lambda3PHExpcv=nEr./(val_PHIS2cv*lambda2);
        YtsPH2cv=exprnd(1/lambda3PHExpcv);
        L_PHIS2cv = (lambda2/lambda3PHExpcv).*exp(-(lambda2-lambda3PHExpcv).*YtsPH2cv);
        Y_AtsPHIS2cv(i)=(R5*gamcdf(val_PHIS2cv/YtsPH2cv,nEr,1/lambda1,'upper')-(R5-1/(1-rho)).*gamcdf(gammaU(j)/YtsPH2cv,nEr,1/lambda1,'upper'))*L_PHIS2cv;
    end 
    % Conditioning on Erlang
    ell_A1PHIS2cv(j)=mean(Y_AtsPHIS2cv)*rho;
    RE_A1PHIS2cv(j)=std(Y_AtsPHIS2cv)/sqrt(N)/ell_A1PHIS2cv(j);
    LE_A1PHIS2cv(j)=log(var(Y_AtsPHIS2cv))/log(ell_A1PHIS2cv(j)^2);
end
figure(1)
semilogy(gammaU,ell_A1PH,'go')
hold on
semilogy(gammaU,ell_A1PHcv,'r*')
hold on
semilogy(gammaU,ell_A1PHIS2,'k.')
hold on
semilogy(gammaU,ell_A1PHIS2cv,'cd')
hold off
set(gca,'fontsize',12.25)

figure(2)
plot(gammaU,RE_A1PH,'go')
hold on
plot(gammaU,RE_A1PHcv,'r*')
hold on
plot(gammaU,RE_A1PHIS2,'k.')
hold on
plot(gammaU,RE_A1PHIS2cv,'cd')
hold off
xlabel('u','FontSize',12)
% ylim([0 0.05])
set(gca,'fontsize',12.25)

figure(3)
plot(gammaU,LE_A1PH,'go')
hold on
plot(gammaU,LE_A1PHcv,'r*')
hold on
plot(gammaU,LE_A1PHIS2,'k.')
hold on
plot(gammaU,LE_A1PHIS2cv,'cd')
hold off
xlabel('u','FontSize',12)
%ylim([0 1])
set(gca,'fontsize',12.25)