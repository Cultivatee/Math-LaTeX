clear; clc; close all;
%数据环境导入
maint_file = '附件2整合数据.xlsx'; 
maint = detectImportOptions(maint_file);
maint = setvartype(maint, {'ID','DATE'}, 'string');
maint_data = readtable(maint_file, maint);
maint_data.DATE = datetime(maint_data.DATE, 'InputFormat', 'yyyy-MM-dd');
maint_data.TypeNum = zeros(height(maint_data),1);
maint_data.TypeNum(contains(maint_data.LEIBIE,'中')) = 1;
maint_data.TypeNum(contains(maint_data.LEIBIE,'大')) = 2;
t_born = datetime(2022,4,1); 
device_ids = unique(maint_data.ID);

%设备辨识绘图
for d = 1:length(device_ids)
    dev = device_ids(d);
    % 实测数据载入
    dev_mod = regexprep(dev,'([A-Za-z])(\d)','$1_$2');
    data_file = sprintf('设备%s_清洗结果_含缺失填补.xlsx', dev_mod);
    if ~isfile(data_file), continue; end
    data = readtable(data_file);
    data.Time = datetime(data.Time,'InputFormat','yyyy-MM-dd HH:mm:ss');
    [dates_u,~,idx_u] = unique(dateshift(data.Time,'start', 'day'));
    daily_per = accumarray(idx_u, data.final_per,[], @mean);
    daily_per = fillmissing(daily_per,'linear');
    %维护数据提取
    idx_mt = (maint_data.ID == dev);
    m_dates = maint_data.DATE(idx_mt);
    m_types = maint_data.TypeNum(idx_mt);
    [m_dates, s_idx_sort] = sort(m_dates);
    m_types = m_types(s_idx_sort);
    %参数识别确认
    seg_starts = [dates_u(1); m_dates + days(1)];
    seg_ends   = [m_dates - days(1); dates_u(end)];
    slopes = []; mid_months = []; gain_mid = []; gain_big = [];
    for k = 1:length(seg_starts)
        mask = dates_u >= seg_starts(k) & dates_u <= seg_ends(k);
        if sum(mask) < 5, continue; end
        % 分段线性拟合斜率
        t_vec = (0:sum(mask)-1)';
        b = [ones(size(t_vec)), t_vec] \ daily_per(mask);
        slopes(end+1) = b(2);  % 斜率，负值表示下降
        % 段中点月份
        mid_date = seg_starts(k) + days(round(sum(mask)/2));
        mid_months(end+1) = month(mid_date);
        % 维护瞬时增益
        if k > 1
            [~, tidx] = min(abs(dates_u - m_dates(k-1)));
            dp = mean(daily_per(tidx:min(tidx+3, end))) - mean(daily_per(max(1, tidx-3):tidx));
            if m_types(k-1) == 1
                gain_mid(end+1) = dp;
            else
                gain_big(end+1) = dp;
            end
        end
    end
    % 基础劣化率 k0
    k_avg = mean(slopes);
    res_k0 = abs(k_avg);  % 取正值为劣化速率
    % 季节算子 S （春季=3-5，夏季=6-8，秋季=9-11，冬季=12-2）
    S_vals = ones(1,4);
    season_idx = zeros(size(mid_months));
    for i = 1:length(mid_months)
        mv = mid_months(i);
        if ismember(mv,[3 4 5])
            season_idx(i) = 1;
        elseif ismember(mv,[6 7 8])
            season_idx(i) = 2;
        elseif ismember(mv,[9 10 11])
            season_idx(i) = 3;
        else
            season_idx(i) = 4;
        end
    end
    for s = 1:4
        if any(season_idx == s)
            S_vals(s) = mean(slopes(season_idx == s)) / k_avg;
        end
    end
    % 链式损伤 Alpha
    a_mid = []; a_big = [];
    for i = 2:length(slopes)
        kp_prev = abs(slopes(i-1) / S_vals(season_idx(i-1))); 
        kp_curr = abs(slopes(i)   / S_vals(season_idx(i)));  
        if kp_prev > 1e-6
            alpha = (kp_curr - kp_prev) / kp_prev;
            if m_types(i-1) == 1
                a_mid(end+1) = alpha;
            else
                a_big(end+1) = alpha;
            end
        end
    end
    % 维护增益
    res_gm = mean(gain_mid, 'omitnan');
    res_gb = mean(gain_big, 'omitnan');
    if isnan(res_gm), res_gm = 5; end    % 缺省值保护
    if isnan(res_gb), res_gb = 12; end
    res_gm = max(5, res_gm);   % 保留原有仿真约束
    res_gb = max(12, res_gb);
    % 损伤系数均值
    alpha_mid_mean = mean(a_mid, 'omitnan');
    alpha_big_mean = mean(a_big, 'omitnan');
    if isnan(alpha_mid_mean), alpha_mid_mean = 0; end
    if isnan(alpha_big_mean), alpha_big_mean = 0; end
    % 历史维护次数统计（新增）
    hist_mid = sum(m_types == 1);
    hist_big = sum(m_types == 2);
    %结果输出
    fprintf('结果输出: %s\n', dev);
    fprintf('  1、劣化与损伤系数:\n');
    fprintf(' （1）基础劣化率 k0:      %.5f \n', res_k0);
    fprintf(' （2）中维护损伤 Alpha_M:   %.4f\n', alpha_mid_mean);
    fprintf(' （3） 大维护损伤 Alpha_B:   %.4f \n', alpha_big_mean);
    fprintf('\n 2、维护维护复增益 (Delta P):\n');
    fprintf(' （1）中维护瞬时增益:      +%.2f %\n', res_gm);
    fprintf(' （2） 大维护瞬时增益:      +%.2f %\n', res_gb);
    fprintf('\n 3、季节算子 (S, 基准=1.0):\n');
    fprintf('     春季: %.2f | 夏季: %.2f | 秋季: %.2f | 冬季: %.2f\n', ...
        S_vals(1), S_vals(2), S_vals(3), S_vals(4)); 
    fprintf('\n  >> 历史维护次数:\n');
    fprintf('     中维护(历史): %d 次\n', hist_mid);
    fprintf('     大维护(历史): %d 次\n', hist_big);
    %回溯计算分析
    back_len = days(dates_u(1) - t_born);
    if back_len > 0
        p_back = zeros(back_len, 1);
        temp_P = daily_per(1);
        for i = 1:back_len
            dt = dates_u(1) - days(i); 
            mv = month(dt);
            s_idx = get_s_idx(mv);
            k_t = res_k0 * S_vals(s_idx);
            gain_val = 0;
            m_idx = find(m_dates == dt, 1);
            if ~isempty(m_idx)
                if m_types(m_idx) == 1
                    gain_val = res_gm;
                else
                    gain_val = res_gb;
                end
            end
            temp_P = temp_P + k_t - gain_val;  % 回溯：加衰退量，减增益
            if temp_P > 140
                temp_P = 140 + rand();  % 轻微随机限制
            end
            p_back(i) = temp_P;
        end
        value_series = [flipud(p_back); daily_per];
    else
        value_series = daily_per;
    end
    %寿命预测
    curr_P = daily_per(end);
    curr_date = dates_u(end);
    time_series = (t_born : days(1) : curr_date)';
    h_win = value_series(max(1, end-364):end);
    prev_avg = mean(h_win);
    has_big_config = any(m_types == 2);
    emergency_used = false;
    below_counter = 0;
    is_dead = false;
    % 新增：模拟维护计数器
    sim_mid = 0;
    sim_big = 0;
    while ~is_dead && curr_date < (t_born + years(25))
        curr_date = curr_date + days(1);
        mv = month(curr_date);
        s_idx = get_s_idx(mv);
        k_t = res_k0 * S_vals(s_idx);
        curr_P = curr_P - k_t;   % 每日自然衰退
        % 定期中维护（每60天自动增益）
        if mod(days(curr_date - dates_u(end)), 60) == 0
            curr_P = curr_P + res_gm;
            sim_mid = sim_mid + 1;   % 记录一次模拟中维护
        end
        %滚动年均值计算
        value_series(end+1) = curr_P;
        time_series(end+1) = curr_date;
        h_win(end+1) = curr_P;
        if length(h_win) > 365, h_win(1) = []; end
        curr_avg = mean(h_win);
        % 滚动年均值跌落触发应急大维护
        if (prev_avg >= 37) && (curr_avg < 37)
            if has_big_config && ~emergency_used
                curr_P = curr_P + res_gb;
                emergency_used = true;
                sim_big = sim_big + 1;   % 记录一次模拟大维护
                value_series(end) = curr_P;
                h_win(end) = curr_P;
                curr_avg = mean(h_win);
            end
            below_counter = 1;
        elseif curr_avg < 37
            below_counter = below_counter + 1;
        else
            below_counter = 0;
        end
        
        if below_counter >= 30 || curr_P < 0
            is_dead = true;
        end
        prev_avg = curr_avg;
    end
    % 总维护次数
    total_mid = hist_mid + sim_mid;
    total_big = hist_big + sim_big;
    %寿命仿真结果输出 
    fprintf('\n 4、寿命预测结果\n')
    fprintf(' (1)回溯出厂 P0: %.2f%\n', value_series(1));
    fprintf('     预计报废日期: %s\n', datestr(curr_date, 'yyyy-mm-dd'));
    fprintf('     总运行寿命: %.2f 年\n', days(curr_date - t_born)/365.25);
    % 经济性分析
    purchase_cost = 3000000;          % 设备购买成本 (元)
    mid_cost = 30000;                 % 中维护单次成本 (元/次)
    big_cost = 120000;                % 大维护单次成本 (元/次)
    total_cost = purchase_cost + total_mid * mid_cost + total_big * big_cost;
    life_years = days(curr_date - t_born) / 365.25;
    avg_annual_cost = total_cost / life_years;
    fprintf('\n  5、经济性分析:\n');
    fprintf('（1）过滤器购买成本: %.0f 元\n', purchase_cost);
    fprintf('（2）中维护总次数: %d 次，成本: %.2f 万元\n', total_mid, total_mid * mid_cost / 10000);
    fprintf('（3）大维护总次数: %d 次，成本: %.2f 万元\n', total_big, total_big * big_cost / 10000);
    fprintf('（4）总成本: %.2f 万元\n', total_cost / 10000);
    fprintf('（5）总运行寿命: %.2f 年\n', life_years);
    fprintf('（6）年平均成本: %.2f 元/年\n', avg_annual_cost);
    fprintf('%s\n\n', repmat('=',1,70));
%生命周期图绘制
[~, start_plot_idx] = min(abs(time_series - dates_u(1)));
t_plot = time_series(start_plot_idx:end);
v_plot = value_series(start_plot_idx:end);
N_plot = min(length(t_plot), length(v_plot));
t_plot = t_plot(1:N_plot);
v_plot = v_plot(1:N_plot);
full_mov_avg = movmean(value_series, [364 0], 'omitnan');
ma_plot = full_mov_avg(start_plot_idx:end);
ma_plot = ma_plot(1:N_plot);
figure('Color','w','Name',char(dev)); hold on; grid on;
plot(t_plot, v_plot, 'r-', 'color','#a30543','LineWidth', 0.8, 'DisplayName', '瞬时透水率');
plot(t_plot, ma_plot, 'k-', 'LineWidth', 1.5, 'DisplayName', '365d均值判定线');
yline(37, 'b--', 'LineWidth', 1.2, 'DisplayName', '阈值 37%');  
datetick('x','yy-mm'); ylabel('透水率 (%)'); 
title(['过滤器 ', char(dev), ' 生命周期图 ']);
legend('Location','best');
end
function s = get_s_idx(mv)
    if ismember(mv,[3 4 5])
        s = 1;    % 春
    elseif ismember(mv,[6 7 8])
        s = 2;    % 夏
    elseif ismember(mv,[9 10 11])
        s = 3;    % 秋
    else
        s = 4;    % 冬
    end
end