function [u, u_hat, omega] = VMD(signal, alpha, tau, K, DC, init, tol)
% Variational Mode Decomposition
% Authors: Konstantin Dragomiretskiy and Dominique Zosso
% zosso@math.ucla.edu --- http://www.math.ucla.edu/~zosso
% Initial release 2013-12-12 (c) 2013
%
% Input and Parameters:
% ---------------------
% signal  - the time domain signal (1D) to be decomposed
% alpha   - the balancing parameter of the data-fidelity constraint
% tau     - time-step of the dual ascent ( pick 0 for noise-slack )
% K       - the number of modes to be recovered
% DC      - true if the first mode is put and kept at DC (0-freq)
% init    - 0 = all omegas start at 0
%                    1 = all omegas start uniformly distributed
%                    2 = all omegas initialized randomly
% tol     - tolerance of convergence criterion; typically around 1e-6
%
% Output:
% -------
% u       - the collection of decomposed modes
% u_hat   - spectra of the modes
% omega   - estimated mode center-frequencies
%
% When using this code, please do cite our paper:
% -----------------------------------------------
% K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
% on Signal Processing (in press)
% please check here for update reference: 
%          http://dx.doi.org/10.1109/TSP.2013.2288675



%---------- Preparations

% Period and sampling frequency of input signal
save_T = length(signal);        % 输入信号的长度
fs = 1/save_T;                  % 采样频率，假设信号是周期的

% extend the signal by mirroring，对信号进行镜像扩展
T = save_T;
f_mirror(1:T/2) = signal(T/2:-1:1);     % 存储为原始信号的148：1的位置的数据，原始代码是f_mirror(1:T/2) = signal(T/2:-1:1)
f_mirror(T/2+1:3*T/2) = signal;
f_mirror(3*T/2+1:2*T) = signal(T:-1:T/2+1);
f = f_mirror;

% Time Domain 0 to T (of mirrored signal)，时间域向量
T = length(f);           % 获得镜像扩展后的长度
t = (1:T)/T;             % 时间向量t范围在（0~1），表示信号的相对时间位置

% Spectral Domain discretization，频域离散化
freqs = t-0.5-1/T;   % t-0.5将频率的中心设置为0，使得正频率和负频率对称地分布在中心线的两侧
                     % -1/T是为了什么，没懂,频率范围是-0.5到0.5-1/T
% Maximum number of iterations (if not converged yet, then it won't anyway)
N = 500;

% For future generalizations: individual alpha for each mode
Alpha = alpha*ones(1,K);      % [2500 2500 2500 2500 2500]

% Construct and center f_hat
F = fft(f);             % 对镜像扩展信号进行傅里叶变换
f_hat = fftshift(F);    % 中心化处理的目的是将频谱的零频率部分移到频谱的中心，这样更容易观察频谱的特性
f_hat_plus = f_hat;
f_hat_plus(1:T/2) = 0;  % 将 f_hat_plus 中前一半长度的元素设为0

% matrix keeping track of every iterant // could be discarded for mem
u_hat_plus = zeros(N, length(freqs), K);   % 500×590×5

% Initialization of omega_k，omega_k代表各个模态函数的中心频率
omega_plus = zeros(N, K);      % 500×5，初始化每个模态分量的中心频率
switch init
    case 1
        for i = 1:K
            omega_plus(1,i) = (0.5/K)*(i-1);
        end
    case 2
        omega_plus(1,:) = sort(exp(log(fs) + (log(0.5)-log(fs))*rand(1,K)));
    otherwise
        omega_plus(1,:) = 0;
end

% if DC mode imposed, set its omega to 0
if DC
    omega_plus(1,1) = 0;
end

% start with empty dual variables
lambda_hat = zeros(N, length(freqs));   % 500×590的零矩阵

% other inits
uDiff = tol+eps; % update step，tol是算法容忍度，eps是一个很小的值，确保uDiff足够小
n = 1; % loop counter
sum_uk = 0; % accumulator，累加器



% ----------- Main loop for iterative updates




while ( uDiff > tol &&  n < N ) % not converged and below iterations limit
    
    % update first mode accumulator，更新第一个模态分量
    k = 1;
    sum_uk = u_hat_plus(n,:,K) + sum_uk - u_hat_plus(n,:,1);
    
    % update spectrum of first mode through Wiener filter of residuals
    u_hat_plus(n+1,:,k) = (f_hat_plus - sum_uk - lambda_hat(n,:)/2)./(1+Alpha(1,k)*(freqs - omega_plus(n,k)).^2);
    
    % update first omega if not held at 0
    if ~DC
        omega_plus(n+1,k) = (freqs(T/2+1:T)*(abs(u_hat_plus(n+1, T/2+1:T, k)).^2)')/sum(abs(u_hat_plus(n+1,T/2+1:T,k)).^2);
    end
    
    % update of any other mode
    for k=2:K
        
        % accumulator
        sum_uk = u_hat_plus(n+1,:,k-1) + sum_uk - u_hat_plus(n,:,k);
        
        % mode spectrum
        u_hat_plus(n+1,:,k) = (f_hat_plus - sum_uk - lambda_hat(n,:)/2)./(1+Alpha(1,k)*(freqs - omega_plus(n,k)).^2);
        
        % center frequencies
        omega_plus(n+1,k) = (freqs(T/2+1:T)*(abs(u_hat_plus(n+1, T/2+1:T, k)).^2)')/sum(abs(u_hat_plus(n+1,T/2+1:T,k)).^2);
        
    end
    
    % Dual ascent
    lambda_hat(n+1,:) = lambda_hat(n,:) + tau*(sum(u_hat_plus(n+1,:,:),3) - f_hat_plus);
    
    % loop counter
    n = n+1;
    
    % converged yet?
    uDiff = eps;
    for i=1:K
        uDiff = uDiff + 1/T*(u_hat_plus(n,:,i)-u_hat_plus(n-1,:,i))*conj((u_hat_plus(n,:,i)-u_hat_plus(n-1,:,i)))';
    end
    uDiff = abs(uDiff);
    
end


%------ Postprocessing and cleanup


% discard empty space if converged early
N = min(N,n);
omega = omega_plus(1:N,:);

% Signal reconstruction
u_hat = zeros(T, K);       % 590×5零矩阵
u_hat((T/2+1):T,:) = squeeze(u_hat_plus(N,(T/2+1):T,:));
u_hat((T/2+1):-1:2,:) = squeeze(conj(u_hat_plus(N,(T/2+1):T,:)));
u_hat(1,:) = conj(u_hat(end,:));

u = zeros(K,length(t));

for k = 1:K
    u(k,:)=real(ifft(ifftshift(u_hat(:,k))));
end

% remove mirror part
u = u(:,T/4+1:3*T/4);

% recompute spectrum
clear u_hat;
for k = 1:K
    u_hat(:,k)=fftshift(fft(u(k,:)))';
end

end