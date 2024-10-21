function IFPR_cell = estimate_missing_values(IFPR_pd, T_pd, p, n, k)
    % estimate_missing_values: 填补不完备模糊偏好矩阵的缺失值
    %
    % 输入：
    % - IFPR_pd: 不完备的模糊偏好矩阵 (矩阵形式)
    % - T_pd: 信任矩阵 (矩阵形式)
    % - p: 决策者数量
    % - n: 每个偏好矩阵的选项数量
    % - k: 近邻数量
    %
    % 输出：
    % - IFPR_cell: 填补好缺失值的模糊偏好矩阵 (元胞数组)

    % 将数据转换为元胞数组，每个元胞包含一个偏好矩阵
    IFPR_cell = cell(p, 1);
    for i = 1:p
        IFPR_cell{i} = reshape(IFPR_pd((i-1)*n+1:i*n, :), n, n);
    end

    % 替换 Excel 中的缺失值为 NaN
    for i = 1:p
        IFPR_cell{i}(IFPR_cell{i} == '~') = NaN;
    end

    % 设置 k 近邻
    knn_in = (1:p)';
    knn_in = repmat(knn_in, 1, p);

    % 复制信任矩阵 T_pd，并将对角线元素设为 -1
    T1 = T_pd;
    T1(logical(eye(size(T1)))) = -1;

    % 按照信任值的降序排列，获取每个决策者的 k 近邻
    [~, knn_in1] = sort(T1, 2, 'descend');
    knn_in(:, 2:end) = knn_in1(:, 1:p-1);

    % 初始化模糊偏好矩阵，将缺失值设为 -5
    for i = 1:p
        IFPR_cell{i}(isnan(IFPR_cell{i})) = -5;
    end

    % 迭代填补缺失值
    IFPR_cell = iterative_estimation(IFPR_cell, knn_in, T_pd, p, n, k);

end
