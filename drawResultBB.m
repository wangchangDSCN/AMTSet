close all%关闭所有的Figure窗口 
clear%清除工作空间的所有变量 
clc%清除命令窗口的内容，对工作环境中的全部变量无任何影响 
warning off all;%把warning的显示关掉

addpath('./util');%添加util路径

pathRes = '.\results\results_SRE_CVPR13\';% The folder containing the tracking results包含跟踪结果的文件夹
pathDraw = '.\tmp\imgs\';% The folder that will stores the images with overlaid bounding box该文件夹将存储带有重叠边框的图像

rstIdx = 1;

seqs=configSeqs;%读取测试集序列文件，configSeqs中保存图片位置等信息

trks=configTrackers;%读取跟踪器信息，包括跟踪器程序入口

if isempty(rstIdx)
    rstIdx = 1;
end

LineWidth = 4;%边界框的宽度

plotSetting;

lenTotalSeq = 0;
resultsAll=[];%矩阵或数组为空
trackerNames=[];
for index_seq=1:length(seqs)%对于每个视频，调用不同跟踪器
    seq = seqs{index_seq};%保存第i个视频信息
    seq_name = seq.name;
    
%     fileName = [pathAnno seq_name '.txt'];
%     rect_anno = dlmread(fileName);
    seq_length = seq.endFrame-seq.startFrame+1; %size(rect_anno,1);%视频总帧数
    lenTotalSeq = lenTotalSeq + seq_length;
    
    for index_algrm=1:length(trks)
        algrm = trks{index_algrm};
        name=algrm.name;
        trackerNames{index_algrm}=name;
               
        fileName = [pathRes seq_name '_' name '.mat'];
    
        load(fileName);
        
        res = results{rstIdx};
        
        if ~isfield(res,'type')&&isfield(res,'transformType')
            res.type = res.transformType;
            res.res = res.res';
        end
            
        if strcmp(res.type,'rect')
            for i = 2:res.len
                r = res.res(i,:);
               
                if (isnan(r) | r(3)<=0 | r(4)<=0)
                    res.res(i,:)=res.res(i-1,:);
                    %             results.res(i,:) = [1,1,1,1];
                end
            end
        end

        resultsAll{index_algrm} = res;

    end
        
    nz	= strcat('%0',num2str(seq.nz),'d'); %number of zeros in the name of image
    
    pathSave = [pathDraw seq_name '_' num2str(rstIdx) '/'];
    if ~exist(pathSave,'dir')
        mkdir(pathSave);
    end
    
    for i=1:seq_length%从第十帧开始
        image_no = seq.startFrame + (i-1);
        id = sprintf(nz,image_no);
        fileName = strcat(seq.path,id,'.',seq.ext);
        
        img = imread(fileName);
        
        imshow(img);

        text(10, 15, ['#' id], 'Color','y', 'FontWeight','bold', 'FontSize',24);%图片左上方字体设置
        
        for j=1:length(trks)
            disp(trks{j}.name)            
           
            LineStyle = plotDrawStyle{j}.lineStyle;
            
            switch resultsAll{j}.type
                case 'rect'
                    rectangle('Position', resultsAll{j}.res(i,:), 'EdgeColor', plotDrawStyle{j}.color, 'LineWidth', LineWidth,'LineStyle',LineStyle);
                case 'ivtAff'
                    drawbox(resultsAll{j}.tmplsize, resultsAll{j}.res(i,:), 'Color', plotDrawStyle{j}.color, 'LineWidth', LineWidth,'LineStyle',LineStyle);
                case 'L1Aff'
                    drawAffine(resultsAll{j}.res(i,:), resultsAll{j}.tmplsize, plotDrawStyle{j}.color, LineWidth, LineStyle);                    
                case 'LK_Aff'
                    [corner c] = getLKcorner(resultsAll{j}.res(2*i-1:2*i,:), resultsAll{j}.tmplsize);
                    hold on,
                    plot([corner(1,:) corner(1,1)], [corner(2,:) corner(2,1)], 'Color', plotDrawStyle{j}.color,'LineWidth',LineWidth,'LineStyle',LineStyle);
                case '4corner'
                    corner = resultsAll{j}.res(2*i-1:2*i,:);
                    hold on,
                    plot([corner(1,:) corner(1,1)], [corner(2,:) corner(2,1)], 'Color', plotDrawStyle{j}.color,'LineWidth',LineWidth,'LineStyle',LineStyle);
                case 'SIMILARITY'
                    warp_p = parameters_to_projective_matrix(resultsAll{j}.type,resultsAll{j}.res(i,:));
                    [corner c] = getLKcorner(warp_p, resultsAll{j}.tmplsize);
                    hold on,
                    plot([corner(1,:) corner(1,1)], [corner(2,:) corner(2,1)], 'Color', plotDrawStyle{j}.color,'LineWidth',LineWidth,'LineStyle',LineStyle);
                otherwise
                    disp('The type of output is not supported!')
                    continue;
            end
        end        
        imwrite(frame2im(getframe(gcf)), [pathSave  num2str(i) '.png']);
    end
    clf
end
