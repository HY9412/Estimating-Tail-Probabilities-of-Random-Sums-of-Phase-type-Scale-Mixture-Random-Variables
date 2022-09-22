% Preliminary comparison for the estimators ConAK2,ConAK2+CV, ConAK2+IS and ConAK2+IS+CV for the scaling random variable being Weibull
clear all;clc;
rho = 0.8;                                            % parameter of geometric distribution
nEr = 2; lambda1 = 3;                                 % parameters of Erlang distribution
a = 2;  b = 0.35;                                     % parameter of scaling distribution being Weibull
N1 = 100; N = 10^5;
gammaU = linspace(1,150000,N1);                       % The thresholds

%% The following is ConAK algorithm conditional on Erlang
ell_onPH = nan(1,N1); RE_onPH = nan(1,N1); LE_onPH = nan(1,N1); 
for j=1:length(gammaU)
    Y_Ats = nan(1,N); 
    for i=1:N
        M=1;
        while rand < rho
            M=M+1;
        end
        if M==1
            val1=gammaU(j);
        else
            Y_11=gamrnd(nEr,1/lambda1,1,M-1);
            Y_22=wblrnd(a,b,1,M-1);
            SS=sum(Y_11.*Y_22);MMM=max(Y_11.*Y_22);
            val1=max(MMM,gammaU(j)-SS);
        end
        Yts=gamrnd(nEr,1/lambda1);
        Y_Ats(i)=M*wblcdf(val1/Yts,a,b,'upper');   
    end
    ell_onPH(j) = mean(Y_Ats)*rho;
    RE_onPH(j) = std(Y_Ats)/sqrt(N)/ell_onPH(j);
    LE_onPH(j)= log(var(Y_Ats))/log(ell_onPH(j)^2);
end

%% The following is ConAK algorithm conditional on Erlang + CV
ell_onPHCV = nan(1,N1); RE_onPHCV = nan(1,N1); LE_onPHCV = nan(1,N1); 
for j=1:length(gammaU)
    Y_AtsCV = nan(1,N); 
    for i=1:N
        M=1;
        while rand < rho
            M=M+1;
        end
        if M==1
            val1CV=gammaU(j);
        else
            Y_11CV=gamrnd(nEr,1/lambda1,1,M-1);
            Y_22CV=wblrnd(a,b,1,M-1);
            SSCV=sum(Y_11CV.*Y_22CV);MMMCV=max(Y_11CV.*Y_22CV);
            val1CV=max(MMMCV,gammaU(j)-SSCV);
        end
        YtsCV=gamrnd(nEr,1/lambda1);
        Y_AtsCV(i)=M*wblcdf(val1CV/YtsCV,a,b,'upper')-(M-1/(1-rho)).*wblcdf(gammaU(j)/YtsCV,a,b,'upper');   
    end
    ell_onPHCV(j) = mean(Y_AtsCV)*rho;
    RE_onPHCV(j) = std(Y_AtsCV)/sqrt(N)/ell_onPHCV(j);
    LE_onPHCV(j) = log(var(Y_AtsCV))/log(ell_onPHCV(j)^2);
end

%% The following is ConAK algorithm conditional on Erlang + IS (density selection Erlang)
ell_onPHIS2 = nan(1,N1); RE_onPHIS2 = nan(1,N1); LE_onPHIS2 = nan(1,N1); 
for j=1:length(gammaU)
    Y_Ats2IS = nan(1,N);
    for i=1:N
        M=1;
        while rand < rho
            M=M+1;
        end
        if M==1
            val1IS=gammaU(j);
        else
            Y_11IS=gamrnd(nEr,1/lambda1,1,M-1);
            Y_22IS=wblrnd(a,b,1,M-1);
            SSIS=sum(Y_11IS.*Y_22IS);MMM=max(Y_11IS.*Y_22IS);
            val1IS=max(MMM,gammaU(j)-SSIS);
        end
        lambdaIS2 = nEr*(a*gamma(1+1/b))/val1IS;
        YtsIS=gamrnd(nEr,1/lambdaIS2);
        LogL2 = nEr*log(lambda1)-nEr*log(lambdaIS2)-(lambda1-lambdaIS2)*YtsIS;
        L2 = exp(LogL2);
        Y_Ats2IS(i) = M*wblcdf(val1IS/YtsIS,a,b,'upper')*L2;
    end
    ell_onPHIS2(j) = mean(Y_Ats2IS)*rho;
    RE_onPHIS2(j) = std(Y_Ats2IS)/sqrt(N)/ell_onPHIS2(j);
    LE_onPHIS2(j) = log(var(Y_Ats2IS))/log(ell_onPHIS2(j)^2);
end

%% The following is ConAK algorithm conditional on Erlang + IS + CV(density selection Erlang)
ell_onPHIS2CV = nan(1,N1); RE_onPHIS2CV = nan(1,N1); LE_onPHIS2CV = nan(1,N1); 
for j=1:length(gammaU)
    tic    
    for i=1:N
        M=1;
        while rand < rho
            M=M+1;
        end
        if M==1
            val1ISCV=gammaU(j);
        else
            Y_11ISCV=gamrnd(nEr,1/lambda1,1,M-1);
            Y_22ISCV=wblrnd(a,b,1,M-1);
            SSISCV=sum(Y_11ISCV.*Y_22ISCV);MMMISCV=max(Y_11ISCV.*Y_22ISCV);
            val1ISCV=max(MMMISCV,gammaU(j)-SSISCV);
        end        
        lambdaIS2CV(j) = nEr*(a*gamma(1+1/b))./val1ISCV;    
        YtsISCV=gamrnd(nEr,1/lambdaIS2CV(j));
        LogL2CV = nEr*log(lambda1)-nEr*log(lambdaIS2CV(j))-(lambda1-lambdaIS2CV(j))*YtsISCV;
        L2CV = exp(LogL2CV);        
        Y_Ats2ISCV(i) = (M*wblcdf(val1ISCV/YtsISCV,a,b,'upper')-(M-1/(1-rho)).*wblcdf(gammaU(j)/YtsISCV,a,b,'upper'))*L2CV;
    end
    ell_onPHIS2CV(j) = mean(Y_Ats2ISCV)*rho;
    RE_onPHIS2CV(j) = std(Y_Ats2ISCV)/sqrt(N)/ell_onPHIS2CV(j);
    LE_onPHIS2CV(j) = log(var(Y_Ats2ISCV))/log(ell_onPHIS2CV(j)^2);
end
% Green circles is ConAK2  % Red stars is ConAK2+CV  % Black dots is ConAK2+IS  % Cyan diamond is ConAK2+IS+CV
figure(1)
semilogy(gammaU,ell_onPH,'go')
hold on
semilogy(gammaU,ell_onPHCV,'r*')
hold on
semilogy(gammaU,ell_onPHIS2,'k.')
hold on
semilogy(gammaU,ell_onPHIS2CV,'cd')
hold off
xlabel('u')
set(gca,'fontsize',12.25)

figure(2)
plot(gammaU,RE_onPH,'go')
hold on
plot(gammaU,RE_onPHCV,'r*')
hold on
plot(gammaU,RE_onPHIS2,'k.')
hold on
plot(gammaU,RE_onPHIS2CV,'cd')
hold off
xlabel('u','FontSize',12)
% ylim([0 0.05])
set(gca,'fontsize',12.25)

figure(3)
plot(gammaU,LE_onPH,'go')
hold on
plot(gammaU,LE_onPHCV,'r*')
hold on
plot(gammaU,LE_onPHIS2,'k.')
hold on
plot(gammaU,LE_onPHIS2CV,'cd')
hold off
xlabel('u','FontSize',12)
% ylim([0 1])
set(gca,'fontsize',12.25)
