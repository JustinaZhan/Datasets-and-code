function IFPR_cell = update_missing_values(IFPR_cell, miss_index, k_index, knn_in, T, k)
    % 初始化隶属度矩阵
    num_miss = size(miss_index, 1);
    membership = zeros(num_miss, k);

    % 计算每个缺失值的隶属度
    for i = 1:num_miss
        DM_index = miss_index(i, 1);
        row = miss_index(i, 2);
        column = miss_index(i, 3);
        sum1 = sum(k_index(i, :) == -1);
        sum2 = k - sum1;
        knn_knn_index = knn_in(k_index(i, 1:sum2), 2:k+1);

        for j = 1:sum2
            DM_knnindex = k_index(i, j);
            sim = zeros(1, k);

            for t = 1:k
                neighbor_index = knn_knn_index(j, t);
                M = IFPR_cell{DM_index};
                M(M == -5) = 5;
                M1 = abs(M - IFPR_cell{neighbor_index});

                row_sum = sum(M1(row, :) <= 1);
                column_sum = sum(M1(:, column) <= 1);
                M1(M1 > 1) = 0;
                row_s = sum(M1(row, :));
                column_s = sum(M1(:, column));
                sim(t) = 1 - (row_s + column_s) ./ (row_sum + column_sum);
            end

            M2 = IFPR_cell{DM_knnindex};
            M2(M2 == -5) = 5;
            M3 = abs(M2 - IFPR_cell{DM_index});

            row_sum1 = sum(M3(row, :) <= 1);
            column_sum1 = sum(M3(:, column) <= 1);
            M3(M3 > 1) = 0;
            row_s1 = sum(M3(row, :));
            column_s1 = sum(M3(:, column));

            sim1 = 1 - (row_s1 + column_s1) ./ (row_sum1 + column_sum1);
            sim2_scalar = (sim1 + sum(sim)) / (sum2 + 1);
            membership(i, j) = (T(DM_index, DM_knnindex) + sim2_scalar) / (1 + sim2_scalar);
        end
    end

    % 根据隶属度更新缺失值
    D = sum(membership, 2);
    [~, first_index] = max(D);

    membership1 = membership(first_index, :) / sum(membership(first_index, :));
    DM_index = miss_index(first_index, 1);
    row = miss_index(first_index, 2);
    column = miss_index(first_index, 3);

    updated_value = 0;
    for j = 1:k
        knn_index_value = k_index(first_index, j);
        if knn_index_value ~= -1
            updated_value = updated_value + membership1(j) * IFPR_cell{knn_index_value}(row, column);
        end
    end

    IFPR_cell{DM_index}(row, column) = updated_value;
    IFPR_cell{DM_index}(column, row) = 1 - updated_value;
end
