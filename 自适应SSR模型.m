%% 自适应SSR算法 - MATLAB版本
% 解决sigma学习率敏感性问题的综合解决方案

clear; clc; close all;

%% 参数设置
N = 63;
T = 3000;
f = 100;
fs = 1000;

%% 运行原始固定sigma算法作为基准
fprintf('=== 运行原始固定sigma算法 ===\n');
[sigma_vals, mse_vals, input_data_for_test] = matlab_style_ssr(N, T, f, fs);
[min_mse_ref, min_idx] = min(mse_vals);
optimal_sigma_ref = sigma_vals(min_idx);

fprintf('参考最优sigma: %.3f, MSE: %.6f\n', optimal_sigma_ref, min_mse_ref);

%% 超参数搜索
[best_params, arm_rewards] = bandit_hyperparameter_search(N, T, input_data_for_test);

%% 使用最佳参数运行完整测试
fprintf('\n=== 使用最佳参数运行完整测试 (T=%d) ===\n', T);
[sigma_hist_best, mse_hist_best] = adaptive_ssr_improved(...
    N, T, input_data_for_test, 0.5, best_params.method, best_params);

%% 比较所有方法
fprintf('\n=== 运行所有方法的比较 ===\n');

% 定义要比较的方法
methods = {};
methods{1} = struct('name', '动量法', 'method', 'momentum', 'alpha', 1e-4, 'momentum', 0.9);
methods{2} = struct('name', 'Adam', 'method', 'adam', 'alpha', 1e-3);
methods{3} = struct('name', '自适应学习率', 'method', 'adaptive', 'alpha_init', 1e-3);
methods{4} = struct('name', 'RMSprop', 'method', 'rmsprop', 'alpha', 1e-3);
methods{5} = struct('name', '最佳参数', 'method', best_params.method);

% 复制最佳参数的所有字段
best_method = methods{5};
fields = fieldnames(best_params);
for i = 1:length(fields)
    if ~strcmp(fields{i}, 'method')
        best_method.(fields{i}) = best_params.(fields{i});
    end
end
methods{5} = best_method;

results = {};
for i = 1:length(methods)
    fprintf('\n测试 %s...\n', methods{i}.name);
    [sigma_hist, mse_hist] = adaptive_ssr_improved(...
        N, T, input_data_for_test, 0.5, methods{i}.method, methods{i});
    
    results{i} = struct(...
        'name', methods{i}.name, ...
        'sigma_hist', sigma_hist, ...
        'mse_hist', mse_hist, ...
        'final_mse', mse_hist(end), ...
        'final_sigma', sigma_hist(end));
    
    fprintf('%s: 最终MSE = %.6f, 最终sigma = %.4f\n', ...
        methods{i}.name, mse_hist(end), sigma_hist(end));
end

%% 可视化结果
figure('Position', [100, 100, 1200, 900]);

% 1. 原始SSR性能曲线
subplot(2,2,1);
plot(sigma_vals, mse_vals, 'k-o', 'LineWidth', 2); hold on;
yline(min_mse_ref, 'r--', sprintf('最优固定Sigma MSE=%.4f', min_mse_ref));
xline(optimal_sigma_ref, 'r--', sprintf('最优固定Sigma=%.3f', optimal_sigma_ref));
title('原始算法性能曲线', 'FontSize', 12);
xlabel('噪声强度 σ');
ylabel('最终MSE');
legend('固定Sigma下的MSE', 'Location', 'best');
grid on;

% 2. Sigma收敛过程比较
subplot(2,2,2);
colors = {'blue', 'green', 'magenta', 'red', 'cyan'};
for i = 1:length(results)
    plot(results{i}.sigma_hist, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('%s (最终: %.3f)', results{i}.name, results{i}.final_sigma));
    hold on;
end
yline(optimal_sigma_ref, 'r--', 'Alpha', 0.5, 'DisplayName', '参考最优');
title('Sigma收敛过程比较');
xlabel('迭代次数');
ylabel('σ');
legend('Location', 'best');
grid on;

% 3. MSE收敛过程比较
subplot(2,2,3);
for i = 1:length(results)
    semilogy(results{i}.mse_hist, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('%s (最终: %.4f)', results{i}.name, results{i}.final_mse));
    hold on;
end
yline(min_mse_ref, 'r--', 'Alpha', 0.5, 'DisplayName', '参考最优');
title('MSE收敛过程比较');
xlabel('迭代次数');
ylabel('MSE');
legend('Location', 'best');
grid on;

% 4. 超参数搜索结果
subplot(2,2,4);
method_names = {'momentum', 'adam', 'adaptive', 'rmsprop', 'scheduled'};
bar(1:length(arm_rewards), arm_rewards, 'FaceAlpha', 0.7);
hold on;
yline(min_mse_ref, 'r--', 'DisplayName', '参考最优');
title('不同优化方法性能比较');
ylabel('平均最终MSE');
set(gca, 'XTickLabel', method_names);
xtickangle(45);
legend('Location', 'best');
grid on;

sgtitle('自适应SSR算法性能比较', 'FontSize', 14, 'FontWeight', 'bold');

%% 最终总结
fprintf('\n=== 最终总结 ===\n');
fprintf('参考固定最优sigma: %.3f, MSE: %.6f\n', optimal_sigma_ref, min_mse_ref);
fprintf('最佳自适应方法: %s, 最终MSE: %.6f\n', best_params.method, results{end}.final_mse);
fprintf('相对于固定最优的性能比: %.2f\n', results{end}.final_mse/min_mse_ref);

%% ================== 函数定义部分 ==================

function [sigma_values, MSE_results, input_data] = matlab_style_ssr(N, T, f, fs, sigma_values)
    % 原始函数：遍历一个固定的sigma列表，为每个sigma计算最终的MSE
    
    if nargin < 5
        sigma_values = [0.01, 0.05, 0.1:0.1:2.0];
    end
    
    ts = 1/fs;
    t = 1:T;
    S_amp = sin(2*pi*f*ts*t);
    input1 = S_amp .* randn(1, T);
    input_data = input1(randperm(T));
    
    MSE_results = zeros(size(sigma_values));
    thetas = zeros(N, 1);
    
    for s = 1:length(sigma_values)
        sigma = sigma_values(s);
        fprintf('测试固定 sigma = %.3f\n', sigma);
        
        noise = sigma * randn(T, N);
        
        % sign 输出强制 ±1，避免 0
        yi = sign(repmat(input_data', 1, N) + noise - repmat(thetas', T, 1));
        yi(yi == 0) = 1; % 强制非零
        yi = yi'; % (N, T)
        
        P = eye(N);
        w = ones(N, 1);
        input_hat = zeros(T, 1);
        
        for j = 1:T
            x = yi(:, j);
            % Kalman-LMS 权重更新
            Px = P * x;
            denom = x' * P * x + 0.5;
            g = Px / denom;
            w = w + g * (input_data(j) - x' * w);
            P = P - g * x' * P;
            input_hat(j) = x' * w;
        end
        
        MSE_results(s) = mean((input_data - input_hat').^2);
    end
end

function [sigma_history, mse_history] = adaptive_ssr_improved(N, T, input_data, sigma_0, method, params)
    % 改进的自适应SSR算法，支持多种学习率优化策略
    
    if length(input_data) < T
        fprintf('警告：输入数据长度(%d)小于所需长度(%d)，将循环使用现有数据\n', ...
            length(input_data), T);
        input_data = repmat(input_data, 1, ceil(T/length(input_data)));
        input_data = input_data(1:T);
    end
    
    thetas = zeros(N, 1);
    
    % 初始化
    P = eye(N);
    w = ones(N, 1);
    sigma = sigma_0;
    
    % 用于记录历史数据
    sigma_history = zeros(T, 1);
    mse_history = zeros(T, 1);
    
    % 不同方法的特定参数初始化
    switch method
        case 'momentum'
            alpha = get_param(params, 'alpha', 1e-4);
            momentum = get_param(params, 'momentum', 0.9);
            velocity = 0;
            fprintf('使用动量法: alpha=%.0e, momentum=%.1f\n', alpha, momentum);
            
        case 'adam'
            alpha = get_param(params, 'alpha', 1e-3);
            beta1 = get_param(params, 'beta1', 0.9);
            beta2 = get_param(params, 'beta2', 0.999);
            epsilon = get_param(params, 'epsilon', 1e-8);
            m = 0; % 一阶动量
            v = 0; % 二阶动量
            fprintf('使用Adam优化器: alpha=%.0e, beta1=%.1f, beta2=%.3f\n', alpha, beta1, beta2);
            
        case 'adaptive'
            alpha_init = get_param(params, 'alpha_init', 1e-3);
            alpha_decay = get_param(params, 'alpha_decay', 0.995);
            alpha_min = get_param(params, 'alpha_min', 1e-6);
            alpha = alpha_init;
            fprintf('使用自适应学习率: alpha_init=%.0e, decay=%.3f\n', alpha_init, alpha_decay);
            
        case 'rmsprop'
            alpha = get_param(params, 'alpha', 1e-3);
            decay_rate = get_param(params, 'decay_rate', 0.9);
            epsilon = get_param(params, 'epsilon', 1e-8);
            cache = 0;
            fprintf('使用RMSprop: alpha=%.0e, decay_rate=%.1f\n', alpha, decay_rate);
            
        case 'scheduled'
            alpha_init = get_param(params, 'alpha_init', 1e-3);
            schedule_type = get_param(params, 'schedule_type', 'exponential');
            fprintf('使用学习率调度: %s\n', schedule_type);
            
        otherwise
            alpha = get_param(params, 'alpha', 5e-5);
            fprintf('使用固定学习率: alpha=%.0e\n', alpha);
    end
    
    fprintf('开始自适应过程... N=%d, T=%d, sigma_0=%.1f\n', N, T, sigma_0);
    
    for j = 1:T
        % 1. 使用当前的sigma生成该时刻的噪声
        noise_j = sigma * randn(N, 1);
        
        % 2. 计算系统输出 - 确保索引在有效范围内
        if j <= length(input_data)
            current_input = input_data(j);
        else
            current_input = input_data(mod(j-1, length(input_data)) + 1);
        end
        
        y_j = sign(current_input + noise_j - thetas);
        y_j(y_j == 0) = 1; % 强制非零
        
        % 3. 计算误差
        error = current_input - y_j' * w;
        
        % 4. 更新权重 w (Kalman-LMS)
        Px = P * y_j;
        denom = y_j' * P * y_j + 0.5;
        g = Px / denom;
        w = w + g * error;
        P = P - g * y_j' * P;
        
        % 5. 计算梯度
        x_k = current_input;
        if sigma < 1e-6
            sigma = 1e-6;
        end
        
        grad_term_multiplier = normpdf(-x_k / sigma) * (-x_k / (sigma^2));
        grad_sigma = error * sum(w * grad_term_multiplier);
        
        % 6. 根据选择的方法更新sigma
        switch method
            case 'momentum'
                velocity = momentum * velocity + alpha * grad_sigma;
                sigma = sigma + velocity;
                
            case 'adam'
                % Adam优化器
                m = beta1 * m + (1 - beta1) * grad_sigma;
                v = beta2 * v + (1 - beta2) * (grad_sigma^2);
                
                % 偏差修正
                m_hat = m / (1 - beta1^j);
                v_hat = v / (1 - beta2^j);
                
                sigma = sigma + alpha * m_hat / (sqrt(v_hat) + epsilon);
                
            case 'adaptive'
                % 自适应学习率：根据梯度大小调整
                if j > 10
                    if abs(grad_sigma) > 1.0
                        alpha = max(alpha * alpha_decay, alpha_min);
                    end
                end
                sigma = sigma + alpha * grad_sigma;
                
            case 'rmsprop'
                % RMSprop
                cache = decay_rate * cache + (1 - decay_rate) * (grad_sigma^2);
                sigma = sigma + alpha * grad_sigma / (sqrt(cache) + epsilon);
                
            case 'scheduled'
                % 学习率调度
                switch schedule_type
                    case 'exponential'
                        current_alpha = alpha_init * (0.95^floor(j/1000));
                    case 'step'
                        current_alpha = alpha_init * (0.5^floor(j/2000));
                    case 'cosine'
                        current_alpha = alpha_init * (1 + cos(pi * j / T)) / 2;
                    otherwise
                        current_alpha = alpha_init;
                end
                sigma = sigma + current_alpha * grad_sigma;
                
            otherwise
                % 默认固定学习率
                sigma = sigma + alpha * grad_sigma;
        end
        
        % 7. 确保 sigma 为正，并添加合理的边界
        sigma = max(min(sigma, 5.0), 1e-6);
        
        % 8. 记录数据
        sigma_history(j) = sigma;
        
        % 计算累积MSE
        input_hat_j = y_j' * w;
        actual_length = min(j, length(input_data));
        if j > 1
            mse_history(j) = mean((input_data(1:actual_length) - ...
                [sigma_history(1:actual_length-1) * 0; input_hat_j]).^2);
        else
            mse_history(j) = error^2;
        end
    end
    
    fprintf('自适应过程结束。最终收敛的 sigma ≈ %.4f\n', sigma);
end

function [best_params, arm_rewards] = bandit_hyperparameter_search(N, T, input_data)
    % 使用多臂赌博机思想搜索最优超参数组合
    
    fprintf('\n=== 开始多臂赌博机超参数搜索 ===\n');
    
    % 定义候选超参数组合
    param_arms = {};
    param_arms{1} = struct('method', 'momentum', 'alpha', 1e-4, 'momentum', 0.9);
    param_arms{2} = struct('method', 'adam', 'alpha', 1e-3, 'beta1', 0.9, 'beta2', 0.999);
    param_arms{3} = struct('method', 'adaptive', 'alpha_init', 1e-3, 'alpha_decay', 0.995);
    param_arms{4} = struct('method', 'rmsprop', 'alpha', 1e-3, 'decay_rate', 0.9);
    param_arms{5} = struct('method', 'scheduled', 'alpha_init', 1e-3, 'schedule_type', 'exponential');
    
    n_trials = 2;
    arm_rewards = zeros(length(param_arms), 1);
    best_reward = inf;
    best_params = [];
    
    for i = 1:length(param_arms)
        params = param_arms{i};
        fprintf('\n测试参数组合 %d: method=%s\n', i, params.method);
        
        % 运行多次试验评估策略性能
        trial_results = zeros(n_trials, 1);
        for trial = 1:n_trials
            [~, mse_hist] = adaptive_ssr_improved(N, T, input_data, 0.5, strategy.method, strategy);
            trial_results(trial) = mse_hist(end);
        end
        
        avg_performance = mean(trial_results);
        method_performances(i) = avg_performance;
        
        fprintf('平均性能指标: %.6f\n', avg_performance);
        
        if avg_performance < best_performance
            best_performance = avg_performance;
            best_params = strategy;
        end
    end
    
    fprintf('\n最优策略配置: method=%s\n', best_params.method);
    fprintf('最佳性能指标: MSE = %.6f\n', best_performance);
end

function value = get_param(params, field_name, default_value)
    % 辅助函数：从参数结构体中获取参数值，如果不存在则使用默认值
    if isfield(params, field_name)
        value = params.(field_name);
    else
        value = default_value;
    end
end