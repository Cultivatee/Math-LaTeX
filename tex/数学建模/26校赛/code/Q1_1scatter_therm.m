%第一问散点图与热力图绘制
clear,clc;
data = readtable('孕周数处理.xlsx');
% 数据获取
week = data{:, 3}; 
BMI = data{:, 4};
YC = data{:, 6};
% Pearson相关系数与显著性p值计算
[r_week, p_week] = corrcoef(week, YC);
[r_BMI, p_BMI] = corrcoef(BMI, YC);
fprintf('孕周数与Y浓度：相关系数 r=%.4f，p值=%.10f\n', r_week(1,2), p_week(1,2));
fprintf('BMI与Y浓度：相关系数 r=%.4f，p值=%.10f\n', r_BMI(1,2), p_BMI(1,2));
% Y染色体浓度与孕周数的关系散点图绘制
figure;
scatter(week, YC);
xlabel('孕周数');
ylabel('Y 染色体浓度');
title('Y 染色体浓度与孕周数的关系散点图');
grid on;
figure;
scatter(BMI, YC);
xlabel('BMI');
ylabel('Y 染色体浓度');
title('Y 染色体浓度与BMI的关系散点图');
grid on;

%Y染色体浓度与孕周数的相关性热力图
[r_w, p_w] = corr(week, YC, 'Type', 'Pearson');

labels_w = {'孕周数', 'Y染色体浓度'};
mat_w  = [1   r_w;
          r_w 1];

figure('Name','Y染色体浓度与孕周数的关系');
h_w = heatmap(labels_w, labels_w, mat_w, ...
              'Title', sprintf('Y染色体浓度与孕周数的关系   r = %.3f', r_w), ...
              'CellLabelColor','k', 'FontSize',14, 'Colormap',parula);
h_w.ColorLimits = [-1,1]; colorbar;

%Y染色体浓度与BMI的相关性热力图
[r_b, p_b] = corr(BMI, YC, 'Type', 'Pearson');

labels_b = {'BMI', 'Y染色体浓度'};
mat_b  = [1   r_b;
          r_b 1];

figure('Name','Y染色体浓度与BMI的关系');
h_b = heatmap(labels_b, labels_b, mat_b, ...
              'Title', sprintf('Y染色体浓度与BMI的关系   r = %.3f, ', r_b), ...
              'CellLabelColor','k', 'FontSize',14, 'Colormap',parula);
h_b.ColorLimits = [-1,1]; colorbar;