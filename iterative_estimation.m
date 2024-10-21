function IFPR_cell = iterative_estimation(IFPR_cell, knn_in, T, p, n, k)
    % 迭代填补缺失值
    while any(cellfun(@(x) any(x(:) < 0), IFPR_cell))

        % 计算缺失值的数量并存储位置
        num_miss = sum(cellfun(@(x) sum(x(:) < 0), IFPR_cell)) / 2;
        miss_index = ones(num_miss, 3);
        index1 = [];
        for i = 1:p
            [row, col] = find(IFPR_cell{i} < 0);
            for j = 1:length(row)
                if row(j) < col(j)
                    index1 = [index1; i, row(j), col(j)];
                end
            end
        end

        index1 = sortrows(index1, [1, 2, 3]);
        ii = 1;
        for i = 1:num_miss
            while index1(ii, 2) > index1(ii, 3)
                ii = ii + 1;
            end
            miss_index(i, :) = index1(ii, :);
            ii = ii + 1;
        end

        % 获取每个缺失值的 k 近邻索引
        k_index = -1 * ones(num_miss, k);
        for i = 1:num_miss
            judge_k = 1;
            for j = 2:p
                judge_index = [knn_in(miss_index(i, 1), j), miss_index(i, 2), miss_index(i, 3)];
                if ismember(judge_index, miss_index, 'rows')
                    continue;
                else
                    k_index(i, judge_k) = knn_in(miss_index(i, 1), j);
                    judge_k = judge_k + 1;
                end
                if judge_k > k
                    break;
                end
            end
        end

        % 计算隶属度并更新缺失值
        IFPR_cell = update_missing_values(IFPR_cell, miss_index, k_index, knn_in, T, k);

    end
end