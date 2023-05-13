% Efficient Parameter Optimization of MCC, this paper has been accepted on IEEE SPL 
% Please cite this paper "An Efficient Parameter Optimization of Maximum Correntropy Criterion" if you use this code.
% evaluation of the proposed algorirthm for different \alpha (coding by Long)

clc; 
clear all; 
close all; 

% setting of unknown system 
unknown_w1 = randn(1,128);
unknown_w2 = -unknown_w1;

% ………………parameters setting……………………
K = 100; % number of trials
L = 128; % filter length
N = 40000; % samples
n = 1:N; 
AR = [1 -0.5]; % AR input with a pole at 0.5
% reset mechanism
V_T = 3*L;
V_D = 0.75*V_T;
diag_M = diag([ones(1,V_T-V_D) zeros(1,V_D)]);
MSD_sum = zeros(4,N); % MSD for different \alpha

%………………run……………………
for kk = 1:4
% newly proposed algorithm
p_ini = 20; % initial kernel width
alfa = [0.99 0.995 0.999 0.9995];
mu_new0 = 0.01;
beta = mu_new0;
MSD_new = zeros(1,N);
midvalue_new_store = zeros(1,N);
midvalue_new_store1 = zeros(1,N);
delta_new_saving = zeros(1,N);
for k = 1:K
    % ………………initialization……………………
    w_new = zeros(L,1);
    input = randn(1,N);
    u = filter(1,AR,input); % correlated input signal
    d = [filter(unknown_w1,1,u(1:N/2)) filter(unknown_w2,1,u(N/2+1:N))]; % desired signal
    impulsive = binornd(1,0.05,1,N) * sqrt(1000*var(d)).* randn(1,N); % BG impulsive noise
    noise_var = 10^-2; % background noise
%     v = (rand(1,N) - 0.5)*sqrt(12*noise_var); % Uniform noise
%     v = sqrt(noise_var)*randn(1,N); % Gaussian noise
    %% generate Laplace noise
    b = sqrt(noise_var)/sqrt(2);
    a = rand(1,N)-0.5;
    v = 0 - b*sign(a).*log(1-2*abs(a));
    D = d + v + impulsive; % 加入30db高斯白噪声
    %………………starting update………………
    p_new = p_ini^2;
    e_new = zeros(1,N);
    ctrl_old_nov = 0;
    ctrl_new_nov = 0;
    for i = L:N
        x = u(i:-1:i-L+1)'; % input vector
        e_new(i) = D(i) - x'*w_new;
        
        mid_value_new = (e_new(i)^2-noise_var + e_new(i)^4-2*noise_var*e_new(i)^2+noise_var^2)/(beta*(x'*x)*e_new(i)^2 + beta*(x'*x)*e_new(i)^4);
        %reset mechanism
        if i >= 3*L
           if (-1-e_new(i)^2)/(2*log(mid_value_new))>0
              stack_new(i) = sqrt((-1-e_new(i)^2)/(2*log(mid_value_new)));
           else
              stack_new(i) = 0;
           end
           if mod(i,V_T) == 0
               sort_stack_new = sort(stack_new(i:-1:i-V_T+1));
               ctrl_new_nov = sort_stack_new*diag_M*sort_stack_new'/(V_T - V_D);
           end
        end
        midvalue_new_store(i) = midvalue_new_store(i) + mid_value_new;
        delta_new = (ctrl_new_nov - ctrl_old_nov)/p_new;
        delta_new_saving(i) = delta_new_saving(i)+delta_new;
        if delta_new > 1
           p_new = p_ini^2; 
           w_new = zeros(L,1);
        else
            if (-1-e_new(i)^2)/(2*log(mid_value_new))<0
               p_new = p_new; 
            else
               p_new = alfa(kk)*p_new + (1-alfa(kk))*min([(-1-e_new(i)^2)/(2*log(mid_value_new)) p_new]);
            end
        midvalue_new_store1(i) = midvalue_new_store(i) + (-1-e_new(i)^2)/(2*log(mid_value_new));
        end
        ctrl_old_nov = ctrl_new_nov;
        
        % update step-size
        mu_new = beta*exp(-1/(2*p_new));
        if mu_new < 0.001
           mu_new =  0.001;
        end
        w_new = w_new + mu_new*exp(-e_new(i)^2/(2*p_new))*x*e_new(i);
        if i <= N/2
           MSD_new(i) = MSD_new(i) + (norm(w_new' - unknown_w1)/norm(unknown_w1))^2;
        else
           MSD_new(i) = MSD_new(i) + (norm(w_new' - unknown_w2)/norm(unknown_w2))^2;
        end
    end   
end
MSD_new = MSD_new/K;
MSD_sum(kk,:) = MSD_new;
end
figure(1);
plot(n,10*log10(MSD_sum(1,:)),'g-');
hold on
plot(n,10*log10(MSD_sum(2,:)),'b-');
hold on
plot(n,10*log10(MSD_sum(3,:)),'r-');
hold on
plot(n,10*log10(MSD_sum(4,:)),'c-');
legend("\alpha=0.99","\alpha=0.995","\alpha=0.999","\alpha=0.9995")
xlabel('Iterations');
ylabel('NMSD(dB)');
