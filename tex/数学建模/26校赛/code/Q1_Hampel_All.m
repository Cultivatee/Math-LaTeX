%% 第一问数据清洗处理 
clear; clc; close all;
% 数据导入
filename = '附件1整合数据.xlsx';          % 数据文件
deviceNum = 10;                   % 设备数量
outputFolder = '设备图表输出';     % 图片保存文件夹
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
%跨设备数据存储
allTime        = cell(deviceNum,1);
allPerFinal    = cell(deviceNum,1);
allMaintDates  = cell(deviceNum,1);
allMaintTypes  = cell(deviceNum,1);
deviceNames    = cell(deviceNum,1);

% 利用Hampel滤波器进行异常值处理
function is_out = adaptive_hampel(per_vec, time_vec, is_exempt, ...
                                   minPts, maxHrs, thr)
    is_out = false(size(per_vec));
    maxDur = hours(maxHrs);
    for i = 1:length(per_vec)
        if isnan(per_vec(i)) || is_exempt(i), continue; end
        t_i = time_vec(i);
        % 向前收集点
        leftIdx = [];
        for j = i-1 : -1 : 1
            if t_i - time_vec(j) > maxDur, break; end
            if ~isnan(per_vec(j)) && ~is_exempt(j)
                leftIdx = [leftIdx, j]; %#ok<AGROW>
                if length(leftIdx) >= minPts, break; end
            end
        end
        % 向后收集点
        rightIdx = [];
        for j = i+1 : length(per_vec)
            if time_vec(j) - t_i > maxDur, break; end
            if ~isnan(per_vec(j)) && ~is_exempt(j)
                rightIdx = [rightIdx, j]; %#ok<AGROW>
                if length(rightIdx) >= minPts, break; end
            end
        end
        if length(leftIdx) < minPts || length(rightIdx) < minPts
            continue;
        end
        windowPer = per_vec([leftIdx, rightIdx]);
        med = median(windowPer);
        mad_val = median(abs(windowPer - med));
        if mad_val == 0
            score = inf * (per_vec(i) ~= med);
        else
            score = abs(per_vec(i) - med) / mad_val;
        end
        if score > thr, is_out(i) = true; end
    end
end

for dev = 1:deviceNum
    sheetName = ['A_', num2str(dev)];
    % 数据读取
    opts = detectImportOptions(filename, 'Sheet', sheetName);
    opts = setvaropts(opts, 'time', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    data = readtable(filename, opts);

    % 维护日期（F列）和维护类型（G列）
    if ismember('日期', data.Properties.VariableNames)
        maintDateCol = data.('日期');
    else
        maintDateCol = data{:, 6};
    end
    if ismember('维护类型', data.Properties.VariableNames)
        maintTypeCol = data.('维护类型');
    else
        maintTypeCol = data{:, 7};
    end
    validMaint = ~isnat(maintDateCol);
    maintDates = maintDateCol(validMaint);
    maintTypes = maintTypeCol(validMaint);
    maintDates = dateshift(maintDates, 'start', 'day');

    % 透水率时间和值，排序
    time = data.time;
    per = data.per;
    [time, idx] = sort(time);
    per = per(idx);
    % 维护标记
    maintWindow = hours(24);
    is_maint_day = false(size(time));
    is_maint_24h = false(size(time));
    for i = 1:length(time)
        thisDay = dateshift(time(i), 'start', 'day');
        if ismember(thisDay, maintDates)
            is_maint_day(i) = true;
            is_maint_24h(i) = true;
        else
            for d = 1:length(maintDates)
                if time(i) > maintDates(d) && ...
                        time(i) <= (maintDates(d) + days(1) + maintWindow)
                    is_maint_24h(i) = true;
                    break;
                end
            end
        end
    end

    % 初始处理输出
    fprintf('总点数: %d, 维护日: %d, 维护日+24h: %d, 原始缺失: %d\n', ...
        length(per), sum(is_maint_day), sum(is_maint_24h), sum(isnan(per)));

    % 第一轮异常检测
    minPts1 = 5; maxHrs1 = 72; thr1 = 3;
    is_outlier1 = adaptive_hampel(per, time, is_maint_24h, minPts1, maxHrs1, thr1);
    fprintf('第一轮异常点: %d\n', sum(is_outlier1));
    per_temp = per;
    per_temp(is_outlier1) = NaN;
    per_filled1 = fillmissing(per_temp, 'linear', 'SamplePoints', time);
    per_filled1(isnan(per)) = NaN;

    % 第二轮检测（维护后24h窗口）
    minPts2 = 5; maxHrs2 = 72; thr2 = 3;
    is_outlier2 = false(size(per));
    targetIdx = find(is_maint_24h & ~is_maint_day);
    for idx = 1:length(targetIdx)
        i = targetIdx(idx);
        if isnan(per_filled1(i)), continue; end
        t_i = time(i);
        leftIdx = []; rightIdx = [];
        for j = i-1 : -1 : 1
            if t_i - time(j) > hours(maxHrs2), break; end
            if ~isnan(per_filled1(j)) && ~is_maint_day(j)
                leftIdx = [leftIdx, j]; %#ok<AGROW>
                if length(leftIdx) >= minPts2, break; end
            end
        end
        for j = i+1 : length(per_filled1)
            if time(j) - t_i > hours(maxHrs2), break; end
            if ~isnan(per_filled1(j)) && ~is_maint_day(j)
                rightIdx = [rightIdx, j];
                if length(rightIdx) >= minPts2, break; end
            end
        end
        if length(leftIdx) < minPts2 || length(rightIdx) < minPts2
            continue;
        end
        windowPer = per_filled1([leftIdx, rightIdx]);
        med = median(windowPer);
        mad_val = median(abs(windowPer - med));
        if mad_val == 0
            score = inf * (per_filled1(i) ~= med);
        else
            score = abs(per_filled1(i) - med) / mad_val;
        end
        if score > thr2, is_outlier2(i) = true; end
    end
    fprintf('第二轮异常点: %d\n', sum(is_outlier2));

    % 异常点整合
    is_outlier_final = is_outlier1 | is_outlier2;
    fprintf('总异常点: %d\n', sum(is_outlier_final));

    per_clean = per;
    per_clean(is_outlier_final) = NaN;
    maint_nan = isnan(per) & is_maint_24h;
    per_final = fillmissing(per_clean, 'linear', 'SamplePoints', time);
    per_final(maint_nan) = NaN;

    % 保存结果到excel
    time_str = cellstr(datestr(time, 'yyyy-mm-dd HH:MM:SS'));
    result = table(time_str, per, is_maint_24h, is_outlier1, is_outlier2, is_outlier_final, per_final, ...
        'VariableNames', {'Time', 'begin_per', 'maintain_24h', 'outlier1', 'outlier2', 'outlier', 'final_per'});
    excelName = ['设备', sheetName, '_清洗结果_含缺失填补.xlsx'];
    writetable(result, excelName);
    fprintf('清洗结果已保存为 %s\n', excelName);

    % 单设备效果图
    % 图一：原始数据散点图
    figRaw = figure('Name', [sheetName, ' 原始数据散点图'], 'Position', [100 100 1000 500]);
    scatter(time, per, 10, 'filled', 'MarkerFaceColor', '#387b85', 'MarkerEdgeColor', 'none');
    xlabel('时间'); ylabel('透水率 (%)');
    title([sheetName, ' 原始数据散点图']);
    grid on;
    for k = 1:length(maintDates)
        xline(maintDates(k), '--', 'color', '#84c3b7', 'LineWidth', 1.5);
    end
    saveas(figRaw, fullfile(outputFolder, [sheetName, '_原始散点图.png']));
    close(figRaw);

    % 图二：清洗后数据散点图
    figClean = figure('Name', [sheetName, ' 清洗后数据散点图'], 'Position', [100 100 1000 500]);
    cleanIdx = ~isnan(per_final);
    scatter(time(cleanIdx), per_final(cleanIdx), 10, 'filled', 'MarkerFaceColor','#387b85', 'MarkerEdgeColor', 'none');
    xlabel('时间'); ylabel('透水率 (%)');
    title([sheetName, ' 清洗后数据散点图']);
    grid on;
    for k = 1:length(maintDates)
        xline(maintDates(k), '--', 'color', '#84c3b7', 'LineWidth', 1.5);
    end
    saveas(figClean, fullfile(outputFolder, [sheetName, '_清洗后散点图.png']));
    close(figClean);

    % 存储跨设备分析所需数据 
    allTime{dev}       = time;
    allPerFinal{dev}   = per_final;
    allMaintDates{dev} = maintDates;
    allMaintTypes{dev} = maintTypes;
    deviceNames{dev}   = sheetName;
end
disp('全部设备处理完成！');
%%跨设备箱线图：不同维护类型恢复幅度对比
allMid = [];
allBig = [];
for dev = 1:deviceNum
    time       = allTime{dev};
    per_final  = allPerFinal{dev};
    maintDates = allMaintDates{dev};
    maintTypes = allMaintTypes{dev};

    for k = 1:length(maintDates)
        idxBefore = find(time < maintDates(k) & ~isnan(per_final), 1, 'last');
        idxAfter  = find(time > maintDates(k) & ~isnan(per_final), 1, 'first');
        if ~isempty(idxBefore) && ~isempty(idxAfter)
            amp = per_final(idxAfter) - per_final(idxBefore);
            if contains(maintTypes(k), '大')
                allBig = [allBig; amp];
            else
                allMid = [allMid; amp];
            end
        end
    end
end
dataPlot = [allMid; allBig];
grp = [repmat({'中维护'}, length(allMid), 1); repmat({'大维护'}, length(allBig), 1)];
if ~isempty(dataPlot)
    figure('Name','跨设备恢复幅度箱线图','Position',[100 100 800 500]);
    myColors = [hex2dec('80')/255, hex2dec('cb')/255, hex2dec('a4')/255;   % #80cba4
                hex2dec('49')/255, hex2dec('65')/255, hex2dec('b0')/255];  % #4965b0
    h=boxplot(dataPlot, grp, 'Colors', myColors);
    set(h, 'LineWidth', 1.5);
    hold on;
    if ~isempty(allMid)
        scatter(ones(size(allMid)) + 0.1*(rand(size(allMid))-0.5), allMid, ...
                20, myColors(1,:), 'filled');
    end
    if ~isempty(allBig)
        scatter(2*ones(size(allBig)) + 0.1*(rand(size(allBig))-0.5), allBig, ...
                20, myColors(2,:), 'filled');
    end
    ylabel('恢复幅度','FontSize', 14);
    title('不同维护类型恢复幅度对比','FontSize', 14);
    set(gca, 'FontSize', 14); 
    grid on;
    saveas(gcf, fullfile(outputFolder, 'Q1_恢复幅度箱线图.png'));
    fprintf('跨设备箱线图已保存至 %s\n', outputFolder);
end