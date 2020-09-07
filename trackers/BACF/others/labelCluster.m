function [ featLabel ] = labelCluster( centers, LAB_feature, N_sample, K )
% function [ featLabel ] sort the input superpixels into K clusters based on
%   the their CIE LAB color features.函数[featLabel]根据输入的超像素的CIE LAB颜色特征将其分类为K个簇。
%
% Input;
%    -centers: formed codebook 
%    -LAB_feature: the LAB features of the boundary superpixels
%    -N_sample: the number of the boundary superpixels
%    -K: the number of clusters
% Output:
%    -featLabel: a row vector of the boundary superpixels' cluster label边界超像素簇标签的行向量


distance = zeros(K,N_sample);

for i=1:K
    for j=1:N_sample
        distance(i,j) = norm(LAB_feature(:,j)-centers(:,i));
    end
end

[minval , featLabel] = min(distance);