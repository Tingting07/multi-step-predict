tic   
clc
clear all
fs=1;                
Ts=1/fs;            
X = xlsread('Yili(process)_month.xlsx');
save origin_data X

L=length(X);         
t=(0:L-1)*Ts;        
STA=0;               
U = X(:,end);

alpha = 2286;       % moderate bandwidth constraint
tau = 0;          % noise-tolerance (no strict fidelity enforcement)
K = 18;              
DC = 0;            
init = 1;           
tol = 1e-7         


[u, u_hat3, omega] = VMD(U, alpha, tau, K, DC, init, tol);

save vmd_data u

figure;
imfn=u;
n=size(imfn,1); 
subplot(n+2,1,1);  
plot(t,X(:,end)); 
ylabel('origin signal','fontsize',12,'FontName','Times New Roman');
title('VMD');

reconstructed_signal = sum(imfn, 1);
for n1=1:n
    subplot(n+2,1,n1+1);
    plot(t,u(n1,:));
    ylabel(['IMF' int2str(n1)]);
end
xlabel('time\itt/hour','fontsize',12,'fontname','Times New Roman');

res = X(:,end)' - reconstructed_signal;
subplot(n+2,1,n+2);
plot(t, res); 
ylabel('res');
xlabel('Êý¾ÝÐòÁÐ','fontsize',12,'fontname','Times New Roman');

