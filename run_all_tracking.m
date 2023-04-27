% ***************************************************************************
% ******   Object Tracking based on all BBO Algorithm   *****
% ***************************************************************************
% used functions :
%                 * rgbHist
%                 * sampling
%                 * norm_win
% ***************************************************************************
%% *** Clear Memory, Screen, and Close Figures ***
clear;
close all;
clc;

%% *** Read sequence of images ***
MetName = 'ICML_mBBO';             % method name

SeqName = 'Woman';  
%SeqName = 'Crossing';                   % sequence name
% To select another sequence, please change this parameter according to the sequence names in the data folder.

SeqPath = ['../data/' SeqName '/' SeqName '/img/'];

SaveFileName = ['Res-' SeqName '-' MetName];

addpath(SeqPath);
img_dir = dir([SeqPath '*.jpg']);
 
num_frm = length(img_dir);                    % number of frames

startFrm = 1;                                     
endFrm = num_frm; 

frm_step = num_frm/3;

first_frm = imread(img_dir(1).name);

%% Read Ground Truth of Sequence
gtName = 'groundtruth_rect.txt';
gtPath = ['../data/' SeqName '/' SeqName '/'];
BoxGT = load([gtPath gtName]);                      % load Ground Truth 
BoxGT = BoxGT(startFrm:endFrm,:);

CLgt = BoxGT(:,1);                                  % coordinates of Ground Truth
RLgt = BoxGT(:,2);
CHgt = CLgt + BoxGT(:,3);
RHgt = RLgt + BoxGT(:,4);

Wgt = CHgt - CLgt;                                  % width of Ground Truth box
Hgt = RHgt - RLgt;                                  % height of Ground Truth box
Agt = Wgt .* Hgt;                                   % Area of Ground Truth box

win_gt = [RLgt CLgt RHgt CHgt];

cent_gt(:,1) = floor((RLgt + RHgt +1)/2);           % center of Ground Truth box
cent_gt(:,2) = floor((CLgt + CHgt +1)/2);           % center of Ground Truth box

%% *** Initialization ***
Ns = 30;                                            % number af particle (population)                                    

rBin = 10;
gBin = 10;
bBin = 10;
Bin = rBin*gBin*bBin;                               % number of histogram columns 

p_u = zeros(Ns,Bin);                                % color histogram model 

SQR = zeros(Ns,Bin);
rho_n = zeros(1,Ns);                                % Bhattacharyya coefficient 
Ln = zeros(1,Ns);
Wn = zeros(num_frm,Ns);

new_cent = zeros(num_frm,2);
old_cent = zeros(num_frm-1,2);

height = size(first_frm,1);
width = size(first_frm,2);

sigm = 0.1;

% BBO Parameters :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MaxIt = 3;                                      % maximum iteration of BBO algorithm 

KeepRate = 0.4;                                 % keep rate
nKeep = round(KeepRate*Ns);                     % Number of Kept Habitats
nNew = Ns - nKeep;                              % Number of New Habitats

mu = linspace(1,0,Ns);                          % emigration rate 
lambda = 1 - mu;                                % immigration rate 

alpha = 0.7;

pMut = 0.2;                                     % mutation rate 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pu_cent = zeros(num_frm,Bin);
SQR_cent = zeros(num_frm,Bin);
rho_cent = zeros(1,num_frm);

Er = zeros(num_frm,1);

Woc = zeros(num_frm,1);
Hoc = zeros(num_frm,1);
Aoc = zeros(num_frm,1);
Aun = zeros(num_frm,1);
Roc = zeros(num_frm,1);

%% *** select tracking window manually ***
GTf = BoxGT(1,:);                                 % first frame ground truth 
cminf = GTf(1,1); 
rminf = GTf(1,2); 
cmaxf = cminf + GTf(1,3); 
rmaxf = rminf + GTf(1,4);

cmin = cminf; cmax = cmaxf; rmin = rminf; rmax = rmaxf;

center(1,1) = floor((rmin + rmax +1)/2);          % the center of window
center(1,2) = floor((cmin + cmax +1)/2);          %

Hy = round(abs(rmax - rmin)/2);                   % half height of window  
Hx = round(abs(cmax - cmin)/2);                   % half width of window
H = [Hy Hx];

new_cent(1,:) = center;

% ***********************************************
% RGB三色图

for r = rminf:rmaxf
    first_frm(r, cminf,:) = [255 0 0];
    first_frm(r, cmaxf,:) = [255 0 0];
end
    
for c = cminf:cmaxf
    first_frm(rminf, c,:) = [255 0 0];
    first_frm(rmaxf, c,:) = [255 0 0];
end



%% ***  Calculate the Target Model ***
q_u = rgbHist(double(first_frm),center,Hx,Hy,rBin,gBin,bBin);

%% *** Do Tracking Algorithm ***
tic
for i = 2:num_frm                                    % number of frames        
    
    old_cent(i-1,:) = new_cent(i-1,:);
    
    framei = imread(img_dir(i).name);
   
    center_Ns = sampling(new_cent(i-1,:),Hx,Hy,Ns);  % propagate particles (habitats)
    
    % calculate Bhattacharyya coefficient for particles 
    for n = 1:Ns                                      
        
        p_u(n,:)= rgbHist(double(framei),center_Ns(n,:),Hx,Hy,rBin,gBin,bBin);
        SQR(n,:) = sqrt(q_u.*p_u(n,:));
        rho_n(1,n) = sum(SQR(n,:));
       
    end
    
    for ii = 1:Ns
       
        par(ii).pos  = center_Ns(ii,:);
        par(ii).cost = rho_n(1,ii);
        
    end    
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % mBBO algorithm
    %{
    for it = 1:MaxIt
        
        [~, SortOrder] = sort([par.cost],'descend');
        par = par(SortOrder);
        BestSol = par(1);
        
        newpar = par;
        
        for ii = 1:Ns
            % Immigration Probabilities
            if rand <= lambda(ii)                   % lambda is zero for first particle
                EP = mu;
                EP(ii) = 0;
                EP = EP/sum(EP);
                
                % Select Source Habitat
                jj = RouletteWheelSelection(EP);
                
                newpar(ii).pos = round(par(ii).pos + ...
                    alpha*(par(jj).pos - par(ii).pos));
                
                p_u(ii,:)= rgbHist(double(framei),newpar(ii).pos,Hx,Hy,rBin,gBin,bBin);   
                SQR(ii,:) = sqrt(q_u.*p_u(ii,:));
                rho_n(1,ii) = sum(SQR(ii,:));
                newpar(ii).cost = rho_n(1,ii);
                
            end
            
            % Mutation
            if rand <= pMut
                DELTA = 0.3*H.*randn(1,2);
                newpar(ii).pos = round(newpar(ii).pos + DELTA);
            
                p_u(ii,:)= rgbHist(double(framei),newpar(ii).pos,Hx,Hy,rBin,gBin,bBin);   
                SQR(ii,:) = sqrt(q_u.*p_u(ii,:));
                rho_n(1,ii) = sum(SQR(ii,:));
                newpar(ii).cost = rho_n(1,ii);
            
            end
 
        end
        
        [~, SortOrder] = sort([newpar.cost],'descend');
        newpar = newpar(SortOrder);
        
        % Select Next Iteration Population
        par = [par(1:nKeep) newpar(1:nNew)];
            
    end
    %}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[par] = tracking_BBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    %[par] = tracking_ICML_BBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    %[par] = tracking_SBBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    %[par] = tracking_ICML_SBBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    %[par] = tracking_DEBBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    %[par] = tracking_ICML_DEBBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    %[par] = tracking_mBBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    [par] = tracking_ICML_mBBO(framei, 30, 1, par, q_u, Hx, Hy, rBin, gBin, bBin);
    
    for kk = 1:Ns
       
        center_Ns(kk,:) = par(kk).pos;
        
    end

    for n = 1:Ns
            
        p_u(n,:)= rgbHist(double(framei),center_Ns(n,:),Hx,Hy,rBin,gBin,bBin);
        
        SQR(n,:) = sqrt(q_u.*p_u(n,:));
        rho_n(1,n) = sum(SQR(n,:));
        Ln(1,n) = (1/sqrt(2*pi*sigm))*exp(-(1-rho_n(1,n))/(2*sigm^2));
     
    end
    
    Wn(i,:) = Ln/sum(Ln);                           % normalize weight of particle
       
    new_cent(i,:) = round(Wn(i,:)*center_Ns);       % new center of object
    
    rmin = round(new_cent(i,1) - Hy);
    rmax = round(new_cent(i,1) + Hy);
    cmin = round(new_cent(i,2) - Hx);
    cmax = round(new_cent(i,2) + Hx);
    
    [rmin,rmax,cmin,cmax] = norm_win(rmin,rmax,cmin,cmax,height,width);
            
    trackImg = framei;  
       
    for r = rmin:rmax
        trackImg(r, cmin,:) = [255 0 0];
        trackImg(r, cmax,:) = [255 0 0];
    end
    
    for c = cmin:cmax
        trackImg(rmin, c,:) = [255 0 0];
        trackImg(rmax, c,:) = [255 0 0];
    end
    
    imshow(trackImg);title([num2str(i),' / ',num2str(num_frm)]);		
	
    if i==2  
        
        imwrite(trackImg,['../results/' SeqName '_#' num2str(floor(i)) '.jpg']);
   
    end
    
    if  mod(i,frm_step)== 0
        index = i/frm_step;
        imwrite(trackImg,['../results/' SeqName '_#' num2str(floor(i)) '.jpg']);
    end

    if i== endFrm 
        
        imwrite(trackImg,['../results/' SeqName '_#' num2str(floor(i)) '.jpg']);
   
    end
    
    hold on; plot(center_Ns(:,2),center_Ns(:,1),'.g','MarkerSize',3);
	pause(.000001);
		
end
toc

%% *** Calculate the Target Area ***
cent_es = new_cent;
RLes = round(new_cent(:,1) - Hy);
CLes = round(new_cent(:,2) - Hx);
RHes = round(new_cent(:,1) + Hy);
CHes = round(new_cent(:,2) + Hx);

win_es = [RLes CLes RHes CHes];

Wes = CHes - CLes;
Hes = RHes - RLes;
Aes = Wes .* Hes;

%% *** Calculate the Center Location Error ***
for  i = 1:num_frm
    
    Er(i,1) = sqrt((cent_es(i,1)-cent_gt(i,1))^2 + (cent_es(i,2)-cent_gt(i,2))^2);
    
end

MeanEr = sum(Er)/num_frm;

%% *** Calculate the Overlap Rate ***
for  i = 1:num_frm
   
    if length(intersect(CLgt(i,1):CHgt(i,1),CLes(i,1):CHes(i,1))) >= 1  &&  length(intersect(RLgt(i,1):RHgt(i,1),RLes(i,1):RHes(i,1))) >= 1
    
        Woc(i,1) = abs(min(CHgt(i,1),CHes(i,1)) - max(CLgt(i,1),CLes(i,1)));
        Hoc(i,1) = abs(min(RHgt(i,1),RHes(i,1)) - max(RLgt(i,1),RLes(i,1)));
        Aoc(i,1) = Woc(i,1)*Hoc(i,1);
    
    else
    
        Aoc(i,1) = 0;
    
    end

    Aun(i,1) = Agt(i,1) + Aes(i,1) - Aoc(i,1);

    Roc(i,1) = Aoc(i,1) / Aun(i,1);
    
end

MeanRoc = sum(Roc)/num_frm;

%% *** Calculate Precision Rate and Success Rate  ***
PR_Tr =[0:50];
for i = 1:length(PR_Tr) 
    
    PR(i,1) = sum(Er <= PR_Tr(i));
    
end
PR = PR ./ num_frm;

SR_Tr = [0:0.01:1];
for i = 1:length(SR_Tr)
    
    SR(i,1) = sum(Roc >= SR_Tr(i));
  
end
SR = SR ./ num_frm;

%% *** Save  Files and Show Results ***
save(['../results/' SaveFileName],'Ns','num_frm','startFrm','endFrm','height','width','BoxGT','cent_gt','win_gt','Wgt','Hgt','Agt','cent_es','win_es','Wes','Hes','Aes','Aoc','Aun','Roc','MeanRoc','Er','MeanEr','PR','SR','PR_Tr','SR_Tr');

figure(2)
plot(1:num_frm,Er(:,1),'-r','LineWidth',1.5,'MarkerSize',5,'MarkerFaceColor','r')
title('Center Location Error')
xlabel('frame number')
ylabel('Error(pixel)')
xlim([0 num_frm]);
saveas(figure(2),['../results/' 'CLE-' SeqName],'jpg')

figure(3)
plot(PR_Tr,PR(:,1),'-','LineWidth',2)
title('Precision Rate Plot')
xlabel('Location Error Threshold')
ylabel('Precision Rate')
saveas(figure(3),['../results/' 'PR-' SeqName],'jpg')


figure(4)
plot(SR_Tr,SR(:,1),'-','LineWidth',2)
title('Success Rate Plot')
xlabel(' Overlap Threshold')
ylabel('Success Rate')
saveas(figure(4),['../results/' 'SR-' SeqName],'jpg')

%%% ***************************************************************************
