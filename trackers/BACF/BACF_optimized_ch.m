% This function implements the BACF tracker.

function [results] = BACF_optimized_ch(params)

%   Setting parameters for local use. 
search_area_scale   = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
nScales             = params.number_of_scales;
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response;

features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);

visualization  = params.visualization;
num_frames     = params.no_fram;
init_target_sz = target_sz;


%set the feature ratio to the feature-cell size 将特征比例设置成特征cell大小
featureRatio = params.t_global.cell_size;    % 4
search_area = prod(init_target_sz / featureRatio * search_area_scale);   %4.3031e+03   搜索区域


% when the number of cells are small, choose a smaller cell size当单元数较少时，请选择较小的单元格大小
if isfield(params.t_global, 'cell_selection_thresh')  
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end

global_feat_params = params.t_global;

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0; 
end

% target size at the initial scale初始规模的目标尺寸
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account窗口大小，考虑填充
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target比例区域，与目标相同的长宽比
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio正方形区域，忽略目标宽高比  repmat填充函数
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding常量填充
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;  %round四舍五入的函数
use_sz = floor(sz/featureRatio);

% construct the label function- correlation output, 2D gaussian function,构造标签函数-相关输出，二维高斯函数
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
% mesh(y);
yf           = fft2(y); %   FFT of y. 

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else 
    interp_sz = use_sz;
end

% construct cosine window构造余弦窗
cos_window = single(hann(use_sz(1))*hann(use_sz(2))');
% mesh(cos_window);
% Calculate feature dimension计算特征尺寸
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
end
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% compute feature dimensionality 计算特征维数
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

if interpolate_response >= 3  
    % Pre-computes the grid that is used for score optimization 预计算用于得分优化的网格
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);    
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end
% initialize the projection matrix (x,y,h,w)初始化投影矩阵
rect_position = zeros(num_frames, 4);
time = 0;

% allocate memory for multi-scale tracking   分配内存以进行多尺度跟踪
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8');
small_filter_sz = floor(base_target_sz/featureRatio);%15 6

loop_frame = 1;
for frame = 1:numel(s_frames)
    %load image载入图片
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end    
    tic();
% i10mage_saliency_1 = demo_BSCA(im);
    %do not estimate translation and scaling on the first frame, since we
    %just want to initialize the tracker there
    %不要估算第一帧的平移和缩放，因为我们只想在那里初始化跟踪器  main loop主循环
    if frame > 1
        for scale_ind = 1:nScales
            multires_pixel_template(:,:,:,scale_ind) = ...  %multires_pixel_template把五个尺度的图合成四维矩阵
                get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);%round四舍五入取整
        end
        xtf = fft2(bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params),cos_window));
        responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);%permute置换数组维度
        
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch如果我们对特征进行了下采样，则希望对响应进行插值，以使其具有与图像补丁相同的大小
        if interpolate_response == 2
            % use dynamic interp size使用动态插入大小
            interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
        end
        responsef_padded = resizeDFT2(responsef, interp_sz);       
        % response in the spatial domain在空间域的响应
        response = ifft2(responsef_padded, 'symmetric');%将responsef_padded视为共轭对称。
        % find maximum peak找到最大的峰值
        if interpolate_response == 3
            error('Invalid parameter value for interpolate_response');
        elseif interpolate_response == 4
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);
        else
            [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
        end
        % calculate translation计算转变翻译
        switch interpolate_response
            case 0
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 1
                translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
            case 2
                translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
            case 3
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 4
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
        end        
        % set the scale设定尺度
        currentScaleFactor = currentScaleFactor * scaleFactors( sind);
        % adjust to make sure we are not to large or to
        % small调整尺度以确保我们不会过大或者过小
        if currentScaleFactor < min_scale_factor
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % update position更新位置
        old_pos = pos;
        pos = pos + translation_vec;
    end
    % extract training sample image region提取训练样本图像区域
    pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz); %patch大小
    [ss,yy,pixels_S] = get_subwindow_no_window(pixels,(use_sz*2),(use_sz));
    mkdir test
    imwrite(pixels_S,'test/pixels_S.jpg');
    S = getSaliencymap(pixels);
    % extract features and do windowing 提取特征
    xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window));
     
    if (frame == 1)
        model_xf = xf;
    else
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
    end
    
    g_f = single(zeros(size(xf)));
%     h_f = g_f;
    h_f = g_f;
    l_f = g_f;
    mu    = 1;
    betha = 4;     %init =10
    mumax = 10000;
    i = 1;
    
    T = prod(use_sz);
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    params.admm_iterations = 2;     %init = 2;
    %   ADMM
    while (i <= params.admm_iterations) 
%           solve for G- please refer to the paper for more details
        B = S_xx + (T * mu);
        S_lx = sum(conj(model_xf) .* l_f, 3);  %conj求共轭
%         S_hx = sum(conj(model_xf) .* h_f, 3);  
        S_hx = sum(conj(model_xf) .* h_f, 3);
%         g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf)) - ((1/mu) * l_f) + h_f) - ...
%             bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);
        g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf)) - ((1/mu) * l_f) + h_f) - ...
            bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);
        %   solve for H   
         h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
         f_s = bsxfun(@times,h,S);
         [sx,sy,f_s] = get_subwindow_no_window(f_s, floor(use_sz/2) , small_filter_sz);
         t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
         t(sx,sy,:) = f_s;
         h_f = fft2(t);
%         h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
%         [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);  %get_subwindow_no_window就是P矩阵，裁剪中间部分
%         t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
%         t(sx,sy,:) = h;
%         h_f = fft2(t);
        
        %   update L
%         l_f = l_f + (mu * (g_f - h_f));
        l_f = l_f + (mu * (g_f - h_f));
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
    end
    
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position and calculate FPS
    rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc();
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            sc_ind = floor((nScales - 1)/2) + 1;
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind))); colormap hsv;
            alpha(resp_handle, 0.2);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(20, 60, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            
            hold off;
        end
        drawnow
    end
    loop_frame = loop_frame + 1;
end
%   save resutls.
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
