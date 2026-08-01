%%第三、四问：维护策略优化模型与参数敏感性分析
clear; clc; close all;
rng(0);   % 固定随机种子
%读取维护数据 
maint_file = '附件2整合数据.xlsx'; 
opts_m = detectImportOptions(maint_file);
opts_m = setvartype(opts_m, {'ID','DATE'}, 'string');
maint_data = readtable(maint_file, opts_m);
maint_data.DATE = datetime(maint_data.DATE, 'InputFormat', 'yyyy-MM-dd');
maint_data.TypeNum = zeros(height(maint_data),1);
maint_data.TypeNum(contains(maint_data.LEIBIE,'中')) = 1;
maint_data.TypeNum(contains(maint_data.LEIBIE,'大')) = 2;
t_born = datetime(2022,4,1); 
device_ids = unique(maint_data.ID);
final_table = [];
dev_params = {};         
%%维护策略优化循环
for d = 1:length(device_ids)
    dev = device_ids(d);
    fprintf('\n正在优化设备 %s...\n', char(dev));
    dev_mod = regexprep(dev,'([A-Za-z])(\d)','$1_$2');
    data_file = sprintf('设备%s_清洗结果_含缺失填补.xlsx', dev_mod);
    if ~isfile(data_file), continue; end
    data = readtable(data_file);
    data.Time = datetime(data.Time,'InputFormat','yyyy-MM-dd HH:mm:ss');
    [dates_u,~,idx_u] = unique(dateshift(data.Time,'start','day'));
    daily_per = accumarray(idx_u, data.final_per,[],@mean);
    daily_per = fillmissing(daily_per,'linear');
    idx_mt = (maint_data.ID == dev);
    m_dates = maint_data.DATE(idx_mt);
    m_types = maint_data.TypeNum(idx_mt);
    [m_dates, s_idx_sort] = sort(m_dates);
    m_types = m_types(s_idx_sort);
    %%参数辨识（继承第二问）
    seg_starts = [dates_u(1); m_dates + days(1)];
    seg_ends   = [m_dates - days(1); dates_u(end)];
    slopes=[]; mid_months=[]; gain_mid=[]; gain_big=[];
    for k=1:length(seg_starts)
        mask = dates_u>=seg_starts(k) & dates_u<=seg_ends(k);
        if sum(mask)<5, continue; end
        t_vec=(0:sum(mask)-1)';
        b=[ones(size(t_vec)),t_vec]\daily_per(mask);
        slopes(end+1)=b(2);
        mid_date = seg_starts(k)+days(round(sum(mask)/2));
        mid_months(end+1)=month(mid_date);
        if k>1
            [~,tidx]=min(abs(dates_u-m_dates(k-1)));
            dp=mean(daily_per(tidx:min(tidx+3,end))) - ...
               mean(daily_per(max(1,tidx-3):tidx));
            if m_types(k-1)==1
                gain_mid(end+1)=dp;
            else
                gain_big(end+1)=dp;
            end
        end
    end
    res_k0 = abs(mean(slopes));
    if isnan(res_k0), res_k0=0.35; end
    %%季节算子
    S_vals = ones(1,4);
    season_idx = zeros(size(mid_months));
    for i=1:length(mid_months)
        mv=mid_months(i);
        season_idx(i)=get_s_idx(mv);
    end
    for s=1:4
        if any(season_idx==s)
            S_vals(s)=mean(slopes(season_idx==s))/mean(slopes);
        end
    end
    %%损伤系数
    a_mid=[]; a_big=[];
    for i=2:length(slopes)
        kp_prev = abs(slopes(i-1)/S_vals(season_idx(i-1)));
        kp_curr = abs(slopes(i)/S_vals(season_idx(i)));
        if kp_prev>1e-6
            alpha=(kp_curr-kp_prev)/kp_prev;
            if m_types(i-1)==1
                a_mid(end+1)=alpha;
            else
                a_big(end+1)=alpha;
            end
        end
    end
    alpha_mid = max(0, mean(a_mid,'omitnan'));
    alpha_big = max(0, mean(a_big,'omitnan'));
    res_gm = max(5, mean(gain_mid,'omitnan'));
    res_gb = max(12, mean(gain_big,'omitnan'));
    %%回溯
    back_len = days(dates_u(1)-t_born);
    temp_P = daily_per(1);
    p_back=zeros(back_len,1);
    for i=1:back_len
        dt=dates_u(1)-days(i);
        s_idx=get_s_idx(month(dt));
        k_t=res_k0*S_vals(s_idx);
        gain=0;
        idx=find(m_dates==dt,1);
        if ~isempty(idx)
            if m_types(idx)==1, gain=res_gm;
            else, gain=res_gb; end
        end
        temp_P=temp_P + k_t - gain;
        temp_P=min(101.5,temp_P);
        p_back(i)=temp_P;
    end
    value_series=[flipud(p_back); daily_per];
    %%GA优化
    lb=[40,150,0.05];
    ub=[120,400,1.5];
    options=optimoptions('ga','PopulationSize',30,...
        'MaxGenerations',25,'Display','off');
    bestX = ga(@(X) fitness_func(X,value_series,dates_u(end),...
        res_k0,S_vals,res_gm,res_gb,...
        alpha_mid,alpha_big,...
        m_types,t_born),3,[],[],[],[],lb,ub,[],options);
    %%获取最优解详情
    [score,L_total,fail_date, v_final] = fitness_func(bestX,value_series,dates_u(end),...
        res_k0,S_vals,res_gm,res_gb,...
        alpha_mid,alpha_big,...
        m_types,t_born);
    %%获取维护次数
    [~, ~, ~, ~, nm, nb] = simulate(bestX, value_series, dates_u(end), ...
        res_k0, S_vals, res_gm, res_gb, alpha_mid, alpha_big, m_types, t_born, ...
        @(nm,nb) 300 + nm*3 + nb*12);
    fprintf('最优策略: 中维护间隔: %d天 大维护间隔: %d天 速率阈值: %.3f\n',...
        round(bestX(1)),round(bestX(2)),bestX(3));
    fprintf('寿命 %.2f 年 | 报废 %s | 年均成本 %.2f | 中维%d次 大维%d次\n',...
        L_total, datestr(fail_date), score, nm, nb);
    final_table=[final_table;
        {char(dev),L_total,score,round(bestX(1)),round(bestX(2)),bestX(3),datestr(fail_date),nm,nb}];
    % 保存基准参数供敏感性分析
    dev_params{d}.device = char(dev);
    dev_params{d}.value_series = value_series;
    dev_params{d}.start_date = dates_u(end);
    dev_params{d}.k0 = res_k0;
    dev_params{d}.S_vals = S_vals;
    dev_params{d}.gm = res_gm;
    dev_params{d}.gb = res_gb;
    dev_params{d}.alpha_m = alpha_mid;
    dev_params{d}.alpha_b = alpha_big;
    dev_params{d}.m_types = m_types;
    dev_params{d}.t_born = t_born;
    dev_params{d}.baseX = bestX;
    dev_params{d}.L_base = L_total;
    dev_params{d}.A0 = score;
    dev_params{d}.nm0 = nm;
    dev_params{d}.nb0 = nb;
    % 绘图 
    t_axis = t_born + days(0:length(v_final)-1);
    idx_plot = find(t_axis >= dates_u(1), 1);
    figure('Name', [char(dev) ' 生命周期仿真'], 'Color', 'w'); hold on; grid on;
    plot(t_axis(idx_plot:end), v_final(idx_plot:end), 'Color', '#a30543', 'LineWidth', 1.2,'DisplayName', '瞬时透水率');
    ma_val = movmean(v_final, [364 0]);
    plot(t_axis(idx_plot:end), ma_val(idx_plot:end), 'Color', '#e6b919', 'LineWidth', 1.2, 'DisplayName', '年均值线');
    yline(37, '--', 'Color','#c9542c', 'Label', '寿命终止阈值(37%)', 'LineWidth', 1,'DisplayName', '寿命终止阈值','LabelHorizontalAlignment', 'left');
    title(['过滤器 ', char(dev), ' 全生命周期透水率变化图']);
    xlabel('时间'); ylabel('透水率 (%)');
    legend('Location', 'southwest');
    datetick('x', 'yyyy-mm');
end
%%输出基准结果
disp('基准优化结果：');
ResultTable = cell2table(final_table,...
    'VariableNames',{'设备','寿命','年均成本','中维护间隔','大维护间隔','速率阈值','报废日期','中维次数','大维次数'});
disp(ResultTable);
save('optimization_results.mat', 'final_table', 'dev_params'); 
fprintf('\n敏感性分析：成本波动下重优化\n');
scenarios = {
    'Baseline',                 1.0,  1.0,  1.0;
    % 购价单因素  ±10%, ±15%
    'Buy +10%',                 1.10, 1.0,  1.0;
    'Buy -10%',                 0.90, 1.0,  1.0;
    'Buy +15%',                 1.15, 1.0,  1.0;
    'Buy -15%',                 0.85, 1.0,  1.0;
    % 大中维护合并：中维护与大维护同步  ±10%, ±15%
    'Mid +10% Big +10%',        1.0,  1.10, 1.10;
    'Mid -10% Big -10%',        1.0,  0.90, 0.90;
    'Mid +15% Big +15%',        1.0,  1.15, 1.15;
    'Mid -15% Big -15%',        1.0,  0.85, 0.85;
    % 全因素同向  ±10%, ±15%
    'All +10%',                 1.10, 1.10, 1.10;
    'All -10%',                 0.90, 0.90, 0.90;
    'All +15%',                 1.15, 1.15, 1.15;
    'All -15%',                 0.85, 0.85, 0.85;
    % 交叉扰动：购价与维护反向
    'Buy+10% Mid-10% Big-10%',  1.10, 0.90, 0.90;
    'Buy-10% Mid+10% Big+10%',  0.90, 1.10, 1.10;
    'Buy+15% Mid-15% Big-15%',  1.15, 0.85, 0.85;
    'Buy-15% Mid+15% Big+15%',  0.85, 1.15, 1.15;
};
n_dev = length(dev_params);
sens_table = [];
options_sens = optimoptions('ga','PopulationSize',20,...
                            'MaxGenerations',15,'Display','off');
lb = [40,150,0.05];
ub = [120,400,1.5];
for d = 1:n_dev
    dev = dev_params{d}.device;
    fprintf('\n----------- Sensitivity for device %s -----------\n', dev);
    VS = dev_params{d}.value_series;
    sd = dev_params{d}.start_date;
    k0 = dev_params{d}.k0;
    S_vals = dev_params{d}.S_vals;
    gm = dev_params{d}.gm;
    gb = dev_params{d}.gb;
    am = dev_params{d}.alpha_m;
    ab = dev_params{d}.alpha_b;
    m_types = dev_params{d}.m_types;
    t_born_dev = dev_params{d}.t_born;
    L0 = dev_params{d}.L_base;
    A0 = dev_params{d}.A0;
    nm0 = dev_params{d}.nm0;
    nb0 = dev_params{d}.nb0;
    for sc = 1:size(scenarios,1)
        label = scenarios{sc,1};
        f_buy = scenarios{sc,2};
        f_mid = scenarios{sc,3};
        f_big = scenarios{sc,4};
        C_fixed = 300*f_buy + nm0*3*f_mid + nb0*12*f_big;
        A_fixed = C_fixed / L0;
        cost_fun = @(nm,nb) 300*f_buy + nm*3*f_mid + nb*12*f_big;
       
        fit_handle = @(X) fitness_cost_only(X, VS, sd, k0, S_vals, gm, gb, am, ab, ...
                                            m_types, t_born_dev, cost_fun);
        bestX_new = ga(fit_handle, 3, [], [], [], [], lb, ub, [], options_sens);
        [A_opt, L_opt, ~, ~, nm_opt, nb_opt] = simulate(bestX_new, VS, sd, ...
            k0, S_vals, gm, gb, am, ab, m_types, t_born_dev, cost_fun);
        delta = (A_fixed - A_opt) / A_opt * 100;
        if abs(delta) <= 3
            feasible = 'Feasible';
        elseif abs(delta) <= 5
            feasible = 'Mild';
        else
            feasible = 'Adjust';
        end
        sens_table = [sens_table; {dev, label, L0, L_opt, A_fixed, A_opt, delta, feasible, ...
                     round(bestX_new(1)), round(bestX_new(2)), bestX_new(3)}];
        fprintf('  %-12s: Afixed=%.2f, Aopt=%.2f, delta=%.2f%%, %s\n', ...
            label, A_fixed, A_opt, delta, feasible);
    end
end
%%函数部分
%适应度函数
function [score, L_total, fail_date, v_out] = fitness_func(X, value_series, start_date, ...
    k0, S_vals, gm, gb, alpha_m, alpha_b, m_types, t_born)
cost_fun = @(nm,nb) 300 + nm*3 + nb*12;
[score, L_total, fail_date, v_out] = simulate(X, value_series, start_date, ...
    k0, S_vals, gm, gb, alpha_m, alpha_b, m_types, t_born, cost_fun);
end
%返回适应度值供GA重优化的函数
function score = fitness_cost_only(X, value_series, start_date, ...
    k0, S_vals, gm, gb, alpha_m, alpha_b, m_types, t_born, cost_fun)
score = simulate(X, value_series, start_date, ...
    k0, S_vals, gm, gb, alpha_m, alpha_b, m_types, t_born, cost_fun);
end
%核心仿真函数
function [score, L_total, fail_date, v_out, nm, nb] = simulate(X, value_series, start_date, ...
    k0, S_vals, gm, gb, alpha_m, alpha_b, m_types, t_born, cost_fun)
T_mid = round(X(1));
T_big = round(X(2));
r_mid = X(3);
curr_P = value_series(end);
curr_date = start_date;
nm = 0; nb = 0;
last_m = 0; last_b = 0;
day = 0;
h_win = value_series(max(1,end-364):end);
prev_avg = mean(h_win);
has_big = any(m_types==2);
emergency = false; below = 0;
while day < 6000
    day = day + 1;
    curr_date = curr_date + days(1);
    s_idx = get_s_idx(month(curr_date));
    D = k0 * S_vals(s_idx) * (1+alpha_m)^nm * (1+alpha_b)^nb;
    curr_P = curr_P - D;
    if length(value_series) > 7
        rate = (curr_P - value_series(end-7)) / 7;
    else
        rate = 0;
    end
    do_big = (day - last_b >= T_big);
    do_mid = ~do_big && ((day - last_m >= T_mid) || (rate <= -r_mid));
    if do_big
        curr_P = curr_P + gb; nb = nb + 1;
        last_b = day; last_m = day;
    elseif do_mid
        curr_P = curr_P + gm; nm = nm + 1;
        last_m = day;
    end
    value_series(end+1) = curr_P;
    h_win(end+1) = curr_P;
    if length(h_win) > 365, h_win(1) = []; end
    curr_avg = mean(h_win);

    if (prev_avg >= 37) && (curr_avg < 37)
        if has_big && ~emergency
            curr_P = curr_P + gb;
            emergency = true;
        end
        below = 1;
    elseif curr_avg < 37
        below = below + 1;
    else
        below = 0;
    end
    if below >= 30 || curr_P <= 0
        break;
    end
    prev_avg = curr_avg;
end
v_out = value_series;
fail_date = curr_date;
L_total = days(curr_date - t_born) / 365.25;
total_cost = cost_fun(nm, nb);
score = total_cost / L_total;
end
%辅助函数
function s = get_s_idx(mv)
if ismember(mv, [3 4 5])
    s = 1;
elseif ismember(mv, [6 7 8])
    s = 2;
elseif ismember(mv, [9 10 11])
    s = 3;
else
    s = 4;
end
end