clc;
clear all;

% 参数设定
p_values = [20, 40, 60, 80, 100];  % 决策者数量的不同取值
n_values = [4, 6, 8, 10, 12];     % 偏好矩阵维度的不同取值
k = 4;  % k近邻数
missing_probs = [0.1, 0.2, 0.3];  % 固定缺失值概率范围
num_trials = 5;  % 每个 p 和 n 下重复实验的次数

% 初始化存储 MSE 的三维数组
mse_values = zeros(length(p_values), length(n_values), length(missing_probs));

% 迭代不同的 p, n 以及缺失值概率
for prob_idx = 1:length(missing_probs)
    missing_prob = missing_probs(prob_idx)  % 当前的缺失值概率
    
    for p_idx = 1:length(p_values)
        p = p_values(p_idx)  % 当前的决策者数量
        
        for n_idx = 1:length(n_values)
            n = n_values(n_idx);  % 当前的偏好矩阵维度
            
            mse_trials = zeros(num_trials, 1);  % 用于存储每次实验的 MSE
            
            % 重复实验 num_trials 次
            for trial = 1:num_trials
                rng(trial);  % 设置随机数种子，每次实验的随机性不同
                
                % 随机生成完整的模糊偏好矩阵（对称位置相加和为1）
                true_IFPR_pd = zeros(p * n, n);
                for i = 1:p
                    % 随机生成一个上三角矩阵
                    M = triu(rand(n, n), 1);  % 生成上三角部分
                    % 确保对称位置相加和为1
                    M = M + tril(1 - M', -1);  % 对称矩阵，并确保M(i,j) + M(j,i) = 1
                    M(logical(eye(n))) = 0.5;  % 对角线元素设置为0.5（可以根据需要调整）
                    true_IFPR_pd((i-1)*n+1:i*n, :) = M;  % 将生成的矩阵存入 true_IFPR_pd
                end

                % 读取信任矩阵
                T_pd = rand(p, p);  % 使用随机生成的信任矩阵作为示例
                T_pd(logical(eye(size(T_pd)))) = 1;  % 对角线为1表示自信任

                % 生成具有缺失值的不完备矩阵
                IFPR_pd = true_IFPR_pd;  % 从实际值复制
                missing_positions = false(size(IFPR_pd));  % 记录缺失位置
                for i = 1:p
                    M = IFPR_pd((i-1)*n+1:i*n, :);  % 取出第i个决策者的偏好矩阵
                    for row = 1:n
                        for col = row+1:n
                            if rand() < missing_prob  % 以指定概率引入缺失值
                                M(row, col) = NaN;
                                M(col, row) = NaN;  % 保持对称位置的NaN
                                missing_positions((i-1)*n+row, col) = true;  % 记录缺失位置
                                missing_positions((i-1)*n+col, row) = true;  % 对称位置也记录
                            end
                        end
                    end
                    IFPR_pd((i-1)*n+1:i*n, :) = M;  % 更新 IFPR_pd
                end

                % 调用估计函数处理缺失值
                IFPR_cell_filled = estimate_missing_values(IFPR_pd, T_pd, p, n, k);

                % 将填补后的元胞数组转换为矩阵
                filled_IFPR_pd = cell2mat(IFPR_cell_filled);

                % 仅计算缺失位置的 MSE
                diff = true_IFPR_pd(missing_positions) - filled_IFPR_pd(missing_positions);  % 误差
                mse_trials(trial) = mean(diff.^2);  % 存储每次实验的 MSE
            end
            
            % 计算 10 次实验的平均 MSE
            mse_values(p_idx, n_idx, prob_idx) = mean(mse_trials);  % 计算平均 MSE
        end
    end
end

% 绘制 1x3 子图的热力图，不显示数字，扩展颜色条范围并添加网格线
figure;
for prob_idx = 1:length(missing_probs)
    subplot(1, 3, prob_idx);
    
   % 绘制热力图，显示单元格内的数字
h = heatmap(n_values, p_values, mse_values(:, :, prob_idx), ...
            'Colormap', parula, 'ColorbarVisible', 'on', 'CellLabelColor', 'black');

    
    % 增加标题
    title(['Missing probability = ', num2str(missing_probs(prob_idx))]);
    
    % 设置颜色范围为 [0.001, 0.2]
    caxis([0.02, 0.15]);
    
    % 添加网格线
    h.GridVisible = 'on';
    
    % 设置轴标签
    xlabel('n');
    ylabel('p');
end

% 调整整体的布局和间距，使得图表更清晰
set(gcf, 'Position', [100, 100, 1200, 200]);  % 调整图形窗口大小

% 输出结果
disp('完成不同 p 和 n 下的 MSE 热力图绘制');  




