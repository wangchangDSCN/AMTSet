clear
close all;
clc

addpath('./util');
addpath('./rstEval');
pathAnno = './anno/';
seqs=configSeqs;
trackers=configTrackers;
numSeq=length(seqs);
numTrk=length(trackers);
thresholdSetcapture=0:1:100;
average_capture_z=0;
capture_length=length(thresholdSetcapture);
evalType = 'AMTset'; 
switch evalType
    case 'SRE'
        rpAll=['.\results\results_SRE_CVPR13\'];
    case {'TRE', 'OPE'}
        rpAll=['.\results\results_TRE_CVPR13\'];
    case 'AMTset'
        rpAll=['.\results\results_TRE_CVPR13\'];
end
A=[];
B={};
% Different thresholds
for tIdx=1:length(thresholdSetcapture)
for idxTrk=1:numTrk
     average_capture_z = 0;
     Totaln=0;
     TotalN=0;
    for idxSeq=1:length(seqs)
    s = seqs{idxSeq};
     
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); 
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,id,'.',s.ext);
    end
    rect_anno = dlmread([pathAnno s.name '.txt']);
    numSeg = 20;
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
    xx=rect_anno(:,1);yy=rect_anno(:,2);zz=rect_anno(:,3);ww=rect_anno(:,4);
     
         N=0;
         m=[];
    for i=1:s.len-1
        a=[xx(i),yy(i)];b=[xx(i+1),yy(i+1)];
       Dtxt=norm(a-b);
       if Dtxt>thresholdSetcapture(tIdx)      
           m(i)=i;                                   
           N=N+1;
           TotalN=TotalN+1;
       end
      m(m==0)=[];
    end
 
     t = trackers{idxTrk};
    b=load([rpAll s.name '_' t.name '.mat']);
    c=b.results{1, 1}.res;
        disp([s.name ' ' t.name]);
        n=0;
        
        coverX=0;
        coverY=0;
        cover=0;
        coverXX=0;
        coverYY=0;
        coverT=0;
        %Judging abrupt motions
        for i=1:N
          L=m(1,i);
          c1=c(L,1);c2=c(L,2);c3=c(L,3);c4=c(L,4);
          d1=rect_anno(L,1);d2=rect_anno(L,2);d3=rect_anno(L,3);d4=rect_anno(L,4);
          c11=c(L+1,1);c22=c(L+1,2);c33=c(L+1,3);c44=c(L+1,4);
          d11=rect_anno(L+1,1);d22=rect_anno(L+1,2);d33=rect_anno(L+1,3);d44=rect_anno(L+1,4);
          % Successful tracking of abrupt motions
          coverX=d3-(d1+d3/2)+(c1+c3/2);
          coverY=c4-(c2+c4/2)+(d2+d4/2);
          cover=coverX*coverY/(c3*c4+d3*d4-coverX*coverY);
          coverXX=d33-(d11+d33/2)+(c11+c33/2);
          coverYY=c44-(c22+c44/2)+(d22+d44/2);
          coverT=coverXX*coverYY/(c33*c44+d33*d44-coverXX*coverYY);
          a=[c11,c22];b=[c1,c2];
          Dmat=norm(a-b);
          if Dmat>thresholdSetcapture(tIdx)&&cover>0.3&&coverT>0.3
              n=n+1;
              Totaln=Totaln+1;
          end
        end
        %AMCR
            average_capture=Totaln/TotalN;
        %Number of abrupt motions captured in AMTSet
             %average_capture=Totaln;
        disp(['capture:',num2str(average_capture)]);     
        disp([trackers{idxTrk}.name]);

    end 
       A(idxTrk,tIdx)=average_capture;
end
end
% figure;
hold on
nameTrk=cell(1,numTrk);
for i=1:length(trackers)
    AA=A(i,:);
    S(1,i)=sum(AA)/capture_length;
    plot(thresholdSetcapture,AA,'LineWidth',4,'color',[0.7 0.7 0.7]);
    t = trackers{i};
    nameTrk(i)=cellstr(t.name);
end

nameTrk=string(nameTrk);
S=string(S);
for i=1:length(trackers)
    zzz(i,:)=strcat(nameTrk(i),{32},S(i));
end
xlabel('Abrupt motion threshold');
ylabel('Abrupt motion capture rate');
title('AMCR plot of AMTset');
legend(strrep(zzz,'_','\_'));
grid;
saveas(gcf,'./perfMat/overall/capture.jpg');
hold off
