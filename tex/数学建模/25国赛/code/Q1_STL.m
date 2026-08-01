%第一问周期性规律、下降趋势分析
clear; clc; close all;
%数据读取
maint_file = '附件2整合数据.xlsx';
if ~isfile(maint_file)
    error('请将附件2整合数据.xlsx放在当前目录');
end
protect = detectImportOptions(maint_file);
protect = setvartype(protect, {'ID'}, 'string');
protect = setvartype(protect, {'DATE'}, 'string');
maint_data = readtable(maint_file, protect);
maint_data.DATE = datetime(maint_data.DATE, 'InputFormat', 'yyyy-MM-dd');
type_num = zeros(height(maint_data),1);
type_num(contains(maint_data.LEIBIE,'中')) = 1;
type_num(contains(maint_data.LEIBIE,'大')) = 2;
maint_data.TypeNum = type_num;
device_ids = unique(maint_data.ID);
fprintf('共 %d 台设备待分析\n\n', length(device_ids));
all_summaries = {};       
all_seg_info  = table();  
mkdir('Figures');         
selected_devs = {'A1','A2','A3','A4','A5'};
season_cell = cell(1,5);        % 每个cell存放 [datenum, seasonal]
season_dev_names = strings(1,5);
season_amp = zeros(1,5);   % 存储对应设备的A_s
%逐台设备进行分析
for d = 1:length(device_ids)
    dev = device_ids(d);
    fprintf('===== 设备 %s =====\n', dev);
    if ~isempty(regexp(dev, '[A-Za-z]\d', 'once'))
        dev_mod = regexprep(dev, '([A-Za-z])(\d)', '$1_$2');
    else
        dev_mod = dev;
    end
    data_file = sprintf('设备%s_清洗结果_含缺失填补.xlsx', dev_mod);
    if ~isfile(data_file)
        warning('   文件 %s 不存在，跳过', data_file);
        continue;
    end
    opts_d = detectImportOptions(data_file, 'Sheet', 'Sheet1');
    opts_d = setvartype(opts_d, {'Time'}, 'string');
    data = readtable(data_file, opts_d);
    if any(strcmp(data.Properties.VariableNames, 'final_per'))
        col_perm = 'final_per';
    elseif any(strcmp(data.Properties.VariableNames, '清洗后透水率'))
        col_perm = '清洗后透水率';
    else
        error('未找到透水率列，请修改代码中的列名');
    end
    data = rmmissing(data, 'DataVariables', {'Time', col_perm});
    data.Time = datetime(data.Time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    data = sortrows(data, 'Time');
    time = data.Time;
    permeability = data.(col_perm);
%设备维护记录导入
    idx_mt = maint_data.ID == dev;
    if sum(idx_mt) == 0
        warning('   设备 %s 无维护记录，跳过', dev);
        continue;
    end
    m_dates = maint_data.DATE(idx_mt);
    m_types = maint_data.TypeNum(idx_mt);
    [m_dates, sort_idx] = sort(m_dates);
    m_types = m_types(sort_idx);
%构建日均透水率，便于分析
    [dates_u, ~, idx_u] = unique(dateshift(time, 'start', 'day'));
    daily_per = accumarray(idx_u, permeability, [], @mean);
    if any(isnan(daily_per))
        daily_per = fillmissing(daily_per, 'linear');
    end
    t_start = dates_u(1);
    t_end = dates_u(end);
    daily_dates = (t_start:days(1):t_end)';
    daily_per_full = interp1(dates_u, daily_per, daily_dates, 'linear', 'extrap');
    % 利用STL进行周期性规律分析
    try
        [trend, seasonal, residual] = trenddecomp(daily_per_full, 'stl', 365);
    catch
        warning('   trenddecomp 不可用，使用经典季节分解');
        [trend, seasonal, residual] = classical_seasonal_decomp(daily_per_full, 365);
    end
    A_s = max(seasonal) - min(seasonal);
    if any(strcmp(dev, selected_devs))
        idx = find(strcmp(selected_devs, dev));
        season_cell{idx} = [datenum(daily_dates), seasonal];
        season_dev_names(idx) = dev;
        season_amp(idx) = A_s;   
    end
    % STL 分解图
    figure('Visible','off');
    figure('Visible','off');
    subplot(4,1,1); plot(daily_dates, daily_per_full, 'color','#f57c6e'); title(sprintf('设备%s 原始日均序列', dev)); grid on;
    subplot(4,1,2); plot(daily_dates, trend, 'color','#71b8ed'); title('趋势项'); grid on;
    subplot(4,1,3); plot(daily_dates, seasonal,'color', '#f57c6e'); title(sprintf('季节项 (振幅=%.2f)', A_s)); grid on;
    subplot(4,1,4); plot(daily_dates, residual,'color', '#b8aeea'); title('残差项'); grid on;
    saveas(gcf, fullfile('Figures', sprintf('%s_STL分解.png', dev)));
    close(gcf);
    % FFT 频谱（去趋势后）
    y_det = detrend(daily_per_full);
    n_y = length(y_det);
    Y = fft(y_det);
    power = abs(Y).^2 / n_y;
    half = floor(n_y/2);
    f_axis = (0:half-1)' * (1/n_y);
    pow_plot = power(1:half);
    [~, i_max] = max(pow_plot(2:end));
    T_max = 1 / f_axis(i_max+1);
    figure('Visible','off');
    plot(f_axis*365, pow_plot); xlabel('频率 (周期/年)'); ylabel('功率');
    title(sprintf('设备%s FFT 功率谱 (主周期≈%.0f天)', dev, T_max)); grid on;
    saveas(gcf, fullfile('Figures', sprintf('%s_FFT频谱.png', dev)));
    close(gcf);
    % ACF 自相关图
    figure('Visible','off');
    autocorr(daily_per_full, 'NumLags',  min(400,n_y-1));
    title(sprintf('设备%s 日均透水率自相关函数', dev));
    saveas(gcf, fullfile('Figures', sprintf('%s_ACF.png', dev)));
    close(gcf);
    % 全局劣化速率
    t = (1:length(daily_per_full))';
    b_glob = [ones(size(t)), t] \ daily_per_full;
    glob_slope_year = b_glob(2) * 365;
    % 分段线性模型构建
    segment_starts = [t_start; m_dates + days(1)];
    segment_ends   = [m_dates - days(1); t_end];
    num_seg = length(segment_starts);
    slopes = []; p_start_fits = []; mid_months = [];
    for k = 1:num_seg
        mask = daily_dates >= segment_starts(k) & daily_dates <= segment_ends(k);
        t_seg = daily_dates(mask); p_seg = daily_per_full(mask);
        if length(t_seg) < 3, continue; end
        days_rel = days(t_seg - segment_starts(k));
        X_seg = [ones(size(days_rel)), days_rel];
        b_seg = X_seg \ p_seg;
        R2 = 1 - sum((p_seg - X_seg*b_seg).^2) / sum((p_seg - mean(p_seg)).^2);
        slopes(end+1) = b_seg(2);
        p_start_fits(end+1) = b_seg(1);
        mid_date = segment_starts(k) + days(round(max(days_rel)/2));
        mid_months(end+1) = month(mid_date);
        % 保存分段信息
        t_row = table({dev}, k, segment_starts(k), segment_ends(k), b_seg(2), R2, b_seg(1), month(mid_date), ...
            'VariableNames', {'Device','Segment','StartDate','EndDate','Slope','R2','P_start','MidMonth'});
        all_seg_info = [all_seg_info; t_row];
    end
    if length(slopes) >= 2
        figure('Visible','off');
        % 左图：斜率随时间散点图 + 线性拟合趋势
        subplot(1,2,1);
        mid_dates_seg = segment_starts(1:length(slopes)) + ...
                        days(round((segment_ends(1:length(slopes)) - segment_starts(1:length(slopes)))/2));
        scatter(mid_dates_seg, slopes, 40, 'filled', 'MarkerFaceColor', [0.2 0.6 0.9]);
        hold on;
        % 线性拟合线
        if length(slopes) >= 3
            p_poly = polyfit(datenum(mid_dates_seg), slopes, 1);
            t_fine = linspace(min(datenum(mid_dates_seg)), max(datenum(mid_dates_seg)), 100);
            plot(datetime(t_fine,'ConvertFrom','datenum'), polyval(p_poly, t_fine), ...
                 'r-', 'color','#f36f43','LineWidth', 1.5);
            title(sprintf('设备%s 分段斜率 (年加速度=%.4f)', dev, p_poly(1)*365));
        else
            title(sprintf('设备%s 分段斜率', dev));
        end
        xlabel('时间'); ylabel('日均劣化斜率 (单位/天)');
        grid on; datetick('x','yyyy-mm','keeplimits');
        
        % 右图：按季节分色的散点图，观察季节内变化
        subplot(1,2,2);
        if ~isempty(slopes)
            % 定义季节颜色
            season_color = zeros(length(slopes),3);
            for si = 1:length(slopes)
                m = mid_months(si);
                if ismember(m,[3 4 5]), season_color(si,:) = [0.2 0.8 0.2];       % 春绿
                elseif ismember(m,[6 7 8]), season_color(si,:) = [1.0 0.5 0.0];    % 夏橙
                elseif ismember(m,[9 10 11]), season_color(si,:) = [0.8 0.2 0.2];  % 秋红
                else, season_color(si,:) = [0.3 0.3 0.8];                         % 冬蓝
                end
            end
            % 横轴用段序号或日期
            x_plot = 1:length(slopes);
            scatter(x_plot, slopes, 40, season_color, 'filled');
            hold on;
            % 连接线以便观察走势
            plot(x_plot, slopes, '-k', 'LineWidth', 0.5);
            xlabel('分段序号'); ylabel('日均劣化斜率');
            title('逐段斜率变化 (颜色:季节)');
            legend({'春','夏','秋','冬'}, 'Location','best');
            grid on;
        end
        saveas(gcf, fullfile('Figures', sprintf('%s_分段斜率演化.png', dev)));
        close(gcf);
    end
    % 分段拟合图
    figure('Visible','off');
    plot(daily_dates, daily_per_full, 'Color', [0.7 0.7 0.7]); hold on;
    for k = 1:num_seg
        if k <= length(slopes) && ~isnan(slopes(k))
            t_line = segment_starts(k):days(1):segment_ends(k);
            days_line = days(t_line - segment_starts(k));
            p_line = p_start_fits(k) + slopes(k) * days_line;
            plot(t_line, p_line, 'r', 'LineWidth', 1.5);
        end
    end
    xline(m_dates, '--k'); legend('日均透水率','分段拟合','维护日期'); grid on;
    title(sprintf('设备%s 分段线性劣化模型', dev));
    saveas(gcf, fullfile('Figures', sprintf('%s_分段拟合.png', dev)));
    close(gcf);

    % 基于斜率构建季节影响算子
    if isempty(slopes)
        base_slope = NaN; sp=NaN; su=NaN; au=NaN; wi=NaN;
        sp_r=NaN; su_r=NaN; au_r=NaN; wi_r=NaN;
    else
        base_slope = mean(slopes);
        season = cell(length(slopes),1);
        for i = 1:length(slopes)
            m = mid_months(i);
            if ismember(m,[3 4 5]), season{i}='春';
            elseif ismember(m,[6 7 8]), season{i}='夏';
            elseif ismember(m,[9 10 11]), season{i}='秋';
            else, season{i}='冬';
            end
        end
        [G_season, s_names] = findgroups(season);
        s_avg = splitapply(@mean, slopes', G_season);
        sp=NaN; su=NaN; au=NaN; wi=NaN;
        for si = 1:length(s_names)
            switch s_names{si}
                case '春', sp = s_avg(si);
                case '夏', su = s_avg(si);
                case '秋', au = s_avg(si);
                case '冬', wi = s_avg(si);
            end
        end
        sp_r = sp / base_slope; su_r = su / base_slope;
        au_r = au / base_slope; wi_r = wi / base_slope;
    end
    % 劣化加速度构建分析下降规律
    if length(slopes) >= 3
        mid_dates_seg = segment_starts(1:length(slopes)) + days(round((segment_ends(1:length(slopes))-segment_starts(1:length(slopes)))/2));
        p_poly = polyfit(datenum(mid_dates_seg), slopes, 1);
        accel_year = p_poly(1) * 365;
    else
        accel_year = NaN;
    end
    % 季节斜率图
    if ~isempty(slopes)
        figure('Visible','off');
        subplot(1,2,1);
        boxchart(categorical(season), slopes);
        ylabel('日均劣化速率'); title('各季节劣化速率分布'); grid on;
        subplot(1,2,2);
        scatter(mid_dates_seg, slopes, 30, 'filled'); hold on;
        if ~isnan(accel_year)
            t_fine = linspace(min(datenum(mid_dates_seg)), max(datenum(mid_dates_seg)), 100)';
            plot(datetime(t_fine,'ConvertFrom','datenum'), polyval(p_poly, t_fine), 'r-', 'LineWidth',1.5);
        end
        xlabel('时间'); ylabel('日均劣化速率');
        title(sprintf('设备%s 劣化速率变化 (年变=%.4f)', dev, accel_year)); grid on;
        saveas(gcf, fullfile('Figures', sprintf('%s_季节斜率.png', dev)));
        close(gcf);
    end
    % 维护顺势增益分析
    before_days = 5; after_days = 60;
    delta_vals = []; dur_vals = [];
    for mi = 1:length(m_dates)
        [~, id_main] = min(abs(daily_dates - m_dates(mi)));
        pre_s = max(1, id_main - before_days); pre_e = id_main - 1;
        post_s = id_main + 1; post_e = min(length(daily_per_full), id_main + after_days);
        if pre_e < pre_s || post_e <= post_s, continue; end
        P_pre = median(daily_per_full(pre_s:pre_e));
        P_post = daily_per_full(post_s:post_e);
        [P_peak, ~] = max(P_post);
        delta_vals(end+1) = P_peak - P_pre;
    end
    mid_mask = (m_types(1:length(delta_vals)) == 1);
    big_mask = (m_types(1:length(delta_vals)) == 2);
    avg_delta_mid = mean(delta_vals(mid_mask), 'omitnan');
    avg_delta_big = mean(delta_vals(big_mask), 'omitnan');

    % 维护瞬时增益图
    figure('Visible','off');
    m_labels = {'中维护','大维护'};
    for t = 1:2
        mask = (m_types(1:length(delta_vals)) == t);
        delta = delta_vals(mask)';
        subplot(1,2,t);
        if sum(mask) >= 2
            boxplot(delta, 'Labels', {m_labels{t}});
        elseif sum(mask) == 1
            bar(delta); set(gca, 'XTickLabel', {m_labels{t}});
            text(1, delta, num2str(delta), 'HorizontalAlignment','center','VerticalAlignment','bottom');
        else
            text(0.5,0.5,'无数据','HorizontalAlignment','center');
        end
        ylabel('ΔP'); title(m_labels{t});
    end
    saveas(gcf, fullfile('Figures', sprintf('%s_维护增益.png', dev)));
    close(gcf);

    % 维护损伤构建
    first_mid_start = NaN; deg_mid = []; deg_big = [];
    for k = 2:num_seg-1
        possible_md = segment_starts(k) - days(1);
        [tf, loc] = ismember(possible_md, m_dates);
        if ~tf || k > length(p_start_fits), continue; end
        m_type = m_types(loc);
        cur_start = p_start_fits(k);
        if k==2 && m_type==1, first_mid_start = cur_start; end
        if ~isnan(first_mid_start) && ~isnan(cur_start)
            deg = first_mid_start - cur_start;
            if m_type==1, deg_mid(end+1)=deg; else, deg_big(end+1)=deg; end
        end
    end
    avg_deg_mid = mean(deg_mid, 'omitnan');
    avg_deg_big = mean(deg_big, 'omitnan');

    % 数据汇总
    summ.Device = dev;
    summ.GlobalSlopeYear = glob_slope_year;
    summ.SeasAmplitude = A_s;           
    summ.BaseSlope = base_slope;
    summ.SpringSlope = sp; summ.SummerSlope = su;
    summ.AutumnSlope = au; summ.WinterSlope = wi;
    summ.SpringRatio = sp_r; summ.SummerRatio = su_r;
    summ.AutumnRatio = au_r; summ.WinterRatio = wi_r;
    summ.DeltaP_Mid = avg_delta_mid; summ.DeltaP_Big = avg_delta_big;
    summ.DamageMid = avg_deg_mid; summ.DamageBig = avg_deg_big;
    summ.AccelYear = accel_year;
    all_summaries{end+1} = summ;

    % 命令行输出
    fprintf('   全局年均劣化速率: %.2f 单位/年\n', glob_slope_year);
    fprintf('   季节振幅(水平): %.2f\n', A_s);
    fprintf('   基准劣化斜率: %.4f 单位/天\n', base_slope);
    fprintf('   季节影响算子: 春 %.2f, 夏 %.2f, 秋 %.2f, 冬 %.2f\n', sp_r, su_r, au_r, wi_r);
    fprintf('   维护增益: 中 %.2f, 大 %.2f\n', avg_delta_mid, avg_delta_big);
    fprintf('   损伤系数: 中 %.2f, 大 %.2f\n', avg_deg_mid, avg_deg_big);
    fprintf('   劣化加速度: %.4f /年\n\n', accel_year);
end

% 绘制A1-A5的季节项分析图
if all(~cellfun(@isempty, season_cell))
    figure('Visible','off');
    t = tiledlayout(5,1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:5
        nexttile;
        data_mat = season_cell{i};
        plot(datetime(data_mat(:,1),'ConvertFrom','datenum'), data_mat(:,2), 'r', 'LineWidth',0.8);
        title(sprintf('设备 %s 季节项 (振幅=%.2f)', season_dev_names(i), season_amp(i)));
        ylabel('透水率');
        if i == 5
            xlabel('日期');
        end
        grid on;
    end
    sgtitle('设备 A1~A5 季节分量对比');
    saveas(gcf, fullfile('Figures', 'A1_A5_Seasonal_Comparison.png'));
    close(gcf);
end
% 设备分段斜率汇总图
if ~isempty(all_seg_info)
    figure('Visible','off');
    devices = unique(string(all_seg_info.Device));
    cmap = lines(length(devices));
    hold on;
    for di = 1:length(devices)
        dev_name = devices(di);                     
        idx_dev = strcmp(all_seg_info.Device, dev_name);  
        dev_slopes = all_seg_info.Slope(idx_dev);
        dev_dates = all_seg_info.StartDate(idx_dev) + ...
                    (all_seg_info.EndDate(idx_dev) - all_seg_info.StartDate(idx_dev))/2;
        h = scatter(dev_dates, dev_slopes, 30, cmap(di,:), 'filled', ...
                    'DisplayName', char(dev_name)); 
        plot(dev_dates, dev_slopes, '-', 'Color', cmap(di,:), 'LineWidth', 0.5, ...
            'HandleVisibility','off');
    end
    xlabel('时间'); ylabel('日均劣化斜率 (单位/天)');
    title('10台设备分段劣化斜率演变');
    legend('show','Location','best');
    grid on;
    datetick('x','yyyy-mm','keeplimits');
    saveas(gcf, fullfile('Figures', 'All_Devices_Slopes_Overview.png'));
    close(gcf);
end
%数据表格汇总
if isempty(all_summaries)
    error('无有效设备分析结果！');
end
T_all = struct2table([all_summaries{:}]);
T_all = sortrows(T_all, 'Device');
disp('10台设备综合指标汇总');
disp(T_all);
writetable(T_all, 'All_Devices_Summary.xlsx');
if ~isempty(all_seg_info)
    writetable(all_seg_info, 'All_Segment_Params.xlsx');
end
% 四季平均斜率汇总表
season_table = T_all(:, {'Device', 'SpringSlope', 'SummerSlope', 'AutumnSlope', 'WinterSlope'});
avg_spring = mean(T_all.SpringSlope, 'omitnan');
avg_summer = mean(T_all.SummerSlope, 'omitnan');
avg_autumn = mean(T_all.AutumnSlope, 'omitnan');
avg_winter = mean(T_all.WinterSlope, 'omitnan');
avg_row = table({'平均'}, avg_spring, avg_summer, avg_autumn, avg_winter, ...
    'VariableNames', season_table.Properties.VariableNames);
season_table = [season_table; avg_row];
writetable(season_table, 'Season_Slopes.xlsx');
disp('四季平均斜率已保存至 Season_Slopes.xlsx');

% 各设备年度劣化斜率汇总表
if ~isempty(all_seg_info)
    mid_dates_all = all_seg_info.StartDate + ...
                    (all_seg_info.EndDate - all_seg_info.StartDate)/2;
    m = month(mid_dates_all);
    y = year(mid_dates_all);
    
    fiscal_start = y;
    fiscal_start(m < 4) = y(m < 4) - 1;
    fiscal_start(fiscal_start >= 2026) = 2025;
    
    % 生成带月份的区间标签
    year_label = strings(size(fiscal_start));
    for i = 1:length(fiscal_start)
        if fiscal_start(i) == 2025
            year_label(i) = "2025.4-2026.4";
        else
            year_label(i) = sprintf("%d.4-%d.3", fiscal_start(i), fiscal_start(i)+1);
        end
    end
    
    % 分组求平均斜率
    dev_str = string(all_seg_info.Device);
    [G, dev_grp, yr_grp] = findgroups(dev_str, year_label);
    avg_slope_yearly = splitapply(@mean, all_seg_info.Slope, G);
    
    devices_unique = unique(dev_str, 'stable');
    all_labels = unique(year_label);
    % 按起始年份排序
    start_years = str2double(extractBefore(all_labels, '.'));
    [~, idx] = sort(start_years);
    years_unique = all_labels(idx);
    
    yearly_matrix = NaN(length(years_unique), length(devices_unique));
    for i = 1:length(dev_grp)
        d_idx = find(strcmp(devices_unique, dev_grp{i}));
        y_idx = find(strcmp(years_unique, yr_grp{i}));
        yearly_matrix(y_idx, d_idx) = avg_slope_yearly(i);
    end
    
    YearlyTable = array2table(yearly_matrix, ...
        'VariableNames', matlab.lang.makeValidName(cellstr(devices_unique)));
    YearlyTable = addvars(YearlyTable, years_unique, 'Before', 1, ...
        'NewVariableNames', {'Year'});
    
    writetable(YearlyTable, 'Yearly_Slopes.xlsx');
    disp('年度劣化斜率（含具体月份区间）已保存至 Yearly_Slopes.xlsx');
end

%辅助函数：经典季节分解
function [trend, seasonal, residual] = classical_seasonal_decomp(x, period)
    n = length(x);
    trend = movmean(x, [period, 0]);
    detrended = x - trend;
    seasonal_pattern = zeros(period,1);
    for i = 1:period
        idx = i:period:n;
        seasonal_pattern(i) = mean(detrended(idx));
    end
    seasonal_pattern = seasonal_pattern - mean(seasonal_pattern);
    seasonal = repmat(seasonal_pattern, ceil(n/period), 1);
    seasonal = seasonal(1:n);
    residual = x - trend - seasonal;
end