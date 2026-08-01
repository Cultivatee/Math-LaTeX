clear; clc; close all;
%数据导入
maint_file = '附件2整合数据.xlsx'; 
maint = detectImportOptions(maint_file);
maint = setvartype(maint, {'ID','DATE'}, 'string');
maint_data = readtable(maint_file, maint);
maint_data.DATE = datetime(maint_data.DATE, 'InputFormat', 'yyyy-MM-dd');
maint_data.TypeNum = zeros(height(maint_data),1);
maint_data.TypeNum(contains(maint_data.LEIBIE,'中')) = 1;
maint_data.TypeNum(contains(maint_data.LEIBIE,'大')) = 2;

device_ids = unique(maint_data.ID);
all_device_params = struct();
all_maint_events = {};  

%设备参数提取
for d = 1:length(device_ids)
    dev = device_ids(d);
    % 数据导入
    dev_mod = regexprep(dev, '([A-Za-z])(\d)', '$1_$2');
    data_file = sprintf('设备%s_清洗结果_含缺失填补.xlsx', dev_mod);
    if ~isfile(data_file), continue; end
    data = readtable(data_file);
    data.Time = datetime(data.Time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    [dates_u, ~, idx_u] = unique(dateshift(data.Time, 'start', 'day'));
    daily_per = accumarray(idx_u, data.final_per, [], @mean);
    daily_per = fillmissing(daily_per, 'linear');
    % 斜率提取
    idx_mt = maint_data.ID == dev;
    m_dates = maint_data.DATE(idx_mt);
    m_types = maint_data.TypeNum(idx_mt);
    [m_dates, s_idx] = sort(m_dates); m_types = m_types(s_idx);
    seg_starts = [dates_u(1); m_dates + days(1)];
    seg_ends   = [m_dates - days(1); dates_u(end)];
    slopes = []; mid_months = []; gain_mid = []; gain_big = [];
    for k = 1:length(seg_starts)
        mask = dates_u >= seg_starts(k) & dates_u <= seg_ends(k);
        if sum(mask) < 5, continue; end
        t_vec = (0:sum(mask)-1)';
        b = [ones(size(t_vec)), t_vec] \ daily_per(mask);
        slopes(end+1) = b(2);
        mid_date = seg_starts(k) + days(round(sum(mask)/2));
        mid_months(end+1) = month(mid_date);
        % 辨识增益 Delta P
        if k > 1
            [~, tidx] = min(abs(dates_u - m_dates(k-1)));
            dp = mean(daily_per(tidx:min(tidx+3, end))) - mean(daily_per(max(1, tidx-3):tidx));
            if m_types(k-1) == 1
                gain_mid(end+1) = dp;
            else
                gain_big(end+1) = dp;
            end
        %维护事件分析
            maint_type = m_types(k-1);          
            maint_month = month(m_dates(k-1));   

            all_maint_events(end+1, :) = {char(dev), maint_type, dp, maint_month};
        end
    end
    %季节算子
    k_avg = mean(slopes);
    S_vals = ones(1,4); 
    season_idx = zeros(size(mid_months));
    for i = 1:length(mid_months)
        m = mid_months(i);
        if ismember(m,[3 4 5]), season_idx(i)=1;
        elseif ismember(m,[6 7 8]), season_idx(i)=2;
        elseif ismember(m,[9 10 11]), season_idx(i)=3;
        else, season_idx(i)=4; end
    end
    for s = 1:4
        if any(season_idx == s), S_vals(s) = mean(slopes(season_idx==s)) / k_avg; end
    end
    
    %链式损伤α
    a_mid = []; a_big = [];
    for i = 2:length(slopes)
        kp_prev = abs(slopes(i-1) / S_vals(season_idx(i-1)));
        kp_curr = abs(slopes(i) / S_vals(season_idx(i)));
        if kp_prev > 1e-6
            alpha = (kp_curr - kp_prev) / kp_prev;
            if m_types(i-1) == 1, a_mid(end+1) = alpha; else, a_big(end+1) = alpha; end
        end
    end

    % 结果输出
    fprintf('结果输出: %s\n', dev);
    fprintf('     基础劣化率 k0:      %.5f \n', abs(k_avg));
    fprintf('     中维护损伤 Alpha_M:   %.4f \n', max(0, mean(a_mid, 'omitnan')));
    fprintf('     大维护损伤 Alpha_B:   %.4f \n', max(0, mean(a_big, 'omitnan')));
    fprintf('     中维护瞬时增益:      +%.2f\n', mean(gain_mid, 'omitnan'));
    fprintf('     大维护瞬时增益:      +%.2f\n', mean(gain_big, 'omitnan'));
    fprintf('     季节算子: 春%.2f 夏%.2f 秋%.2f 冬%.2f\n', S_vals);
    res.k0 = abs(k_avg); res.S = S_vals; 
    res.am = mean(a_mid); res.ab = mean(a_big);
    res.gm = mean(gain_mid); res.gb = mean(gain_big);
    all_device_params.(char(dev)) = res;
end
%数据表格化
devNames = fieldnames(all_device_params);
nDev = length(devNames);
gm_all = NaN(nDev,1); gb_all = NaN(nDev,1);
am_all = NaN(nDev,1); ab_all = NaN(nDev,1);

for i = 1:nDev
    dev = devNames{i};
    gm_all(i) = all_device_params.(dev).gm;
    gb_all(i) = all_device_params.(dev).gb;
    am_all(i) = max(0, all_device_params.(dev).am);
    ab_all(i) = max(0, all_device_params.(dev).ab);
end
diff_gain = gm_all - gb_all;
diff_alpha = am_all - ab_all;

gainTable = table(devNames, gm_all, gb_all, diff_gain, ...
    'VariableNames', {'设备ID', '中维护增益', '大维护增益', '中维护减大维护'});
avg_gm = mean(gm_all, 'omitnan'); avg_gb = mean(gb_all, 'omitnan');
gainTable_avg = table({'平均'}, avg_gm, avg_gb, avg_gm - avg_gb, ...
    'VariableNames', {'设备ID', '中维护增益', '大维护增益', '中维护减大维护'});
gainTable = [gainTable; gainTable_avg];

alphaTable = table(devNames, am_all, ab_all, diff_alpha, ...
    'VariableNames', {'设备ID', '中维护损伤', '大维护损伤', '中维护减小维护'});
avg_am = mean(am_all, 'omitnan'); avg_ab = mean(ab_all, 'omitnan');
alphaTable_avg = table({'平均'}, avg_am, avg_ab, avg_am - avg_ab, ...
    'VariableNames', {'设备ID', '中维护损伤', '大维护损伤', '中维护减小维护'});
alphaTable = [alphaTable; alphaTable_avg];

fprintf('\n%s\n', repmat('=',1,80));
disp('维维护增益数据');
disp(gainTable);
fprintf('\n%s\n', repmat('=',1,80));
disp('维维护损伤数据');
disp(alphaTable);

%数据转化为表格输出
if isempty(all_maint_events)
    error('未收集到任何维护事件，请检查数据！');
end
event_table = cell2table(all_maint_events, ...
    'VariableNames', {'DeviceID', 'TypeNum', 'Gain', 'Month'});
event_table.DeviceID = string(event_table.DeviceID);
% 添加季节标签
season_label = strings(height(event_table), 1);
for i = 1:height(event_table)
    m = event_table.Month(i);
    if ismember(m, [3 4 5])
        season_label(i) = "春";
    elseif ismember(m, [6 7 8])
        season_label(i) = "夏";
    elseif ismember(m, [9 10 11])
        season_label(i) = "秋";
    else
        season_label(i) = "冬";
    end
end
event_table.Season = season_label;
devices = unique(event_table.DeviceID, 'stable');
seasons = ["春", "夏", "秋", "冬"]';
nDev = length(devices);
nSea = length(seasons);
% Sheet1: 全部维护
gain_all = NaN(nDev, nSea);
for i = 1:nDev
    for j = 1:nSea
        idx = event_table.DeviceID == devices(i) & event_table.Season == seasons(j);
        if any(idx), gain_all(i,j) = mean(event_table.Gain(idx), 'omitnan'); end
    end
end
avg_all = mean(gain_all, 1, 'omitnan');
T_all = array2table([gain_all; avg_all], ...
    'RowNames', [cellstr(devices); '平均'], 'VariableNames', cellstr(seasons));
% Sheet2: 中维护
gain_mid = NaN(nDev, nSea);
for i = 1:nDev
    for j = 1:nSea
        idx = event_table.DeviceID == devices(i) & event_table.Season == seasons(j) & event_table.TypeNum == 1;
        if any(idx), gain_mid(i,j) = mean(event_table.Gain(idx), 'omitnan'); end
    end
end
avg_mid = mean(gain_mid, 1, 'omitnan');
T_mid = array2table([gain_mid; avg_mid], ...
    'RowNames', [cellstr(devices); '平均'], 'VariableNames', cellstr(seasons));
% Sheet3: 大维护
gain_big = NaN(nDev, nSea);
for i = 1:nDev
    for j = 1:nSea
        idx = event_table.DeviceID == devices(i) & event_table.Season == seasons(j) & event_table.TypeNum == 2;
        if any(idx), gain_big(i,j) = mean(event_table.Gain(idx), 'omitnan'); end
    end
end
avg_big = mean(gain_big, 1, 'omitnan');
T_big = array2table([gain_big; avg_big], ...
    'RowNames', [cellstr(devices); '平均'], 'VariableNames', cellstr(seasons));
% 转成Excel 
writetable(T_all, '季节维护增益.xlsx', 'Sheet', '全部维护', 'WriteRowNames', true);
writetable(T_mid, '季节维护增益.xlsx', 'Sheet', '中维护', 'WriteRowNames', true);
writetable(T_big, '季节维护增益.xlsx', 'Sheet', '大维护', 'WriteRowNames', true);