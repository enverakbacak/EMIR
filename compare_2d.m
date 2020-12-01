
% https://www.mathworks.com/matlabcentral/answers/361878-getting-the-data-points-of-each-cluster-in-kmeans
close all;
clear all;
clc;

%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/hashCodes_16.mat');
data = hashCodes_16;
%load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/features/features_16.mat');
%features = features_16;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
queryIndex = xlsread('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/qLabels_V2.xls');  % Reads randomly choosen query pairs from excell filequeryIndex = transpose( queryIndex ); 

%}

%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/hashCodes_128.mat');
data = hashCodes_128;
%load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/features/features_16.mat');
%features = features_16;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
queryIndex = xlsread('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/streets_2d.xls');  % Reads randomly choosen query pairs from excell filequeryIndex = transpose( queryIndex ); 

%}
  



load('Lamda/hashCodes/hashCodes_128.mat');
data = hashCodes_128;
data = max(data,0);
%load('Lamda/features/features_16.mat');
%features = features_16;
load('Lamda/hashCodes/targets.mat');
targets = targets;
load('Lamda/hashCodes/filenames.mat');
queryIndex = xlsread('Lamda/qGroups_2d.xls');
filenames = filenames;
N = length(filenames);
%}

%{
load('Barcelona/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
data = max(data,0);
%load('Barcelona/features/features_16.mat');
%features = features_16;
load('Barcelona/hashCodes/targets.mat');
targets = targets;
load('Barcelona/hashCodes/filenames.mat');
queryIndex = xlsread('Barcelona/qGroups_2d.xls');
filenames = filenames;
N = length(filenames);
%}


queryIndex = transpose( queryIndex ); 
queryIndex1 = queryIndex(1,:);        % First element of Query Pair
queryIndex2 = queryIndex(2,:);        % Second element of Query Pair

maxFront = 3;           % tHE NUMBER OF pARETO FRONTS
P = maxFront;           % The number points nearest to MQOP

 for l = 1:500 % 500 FOR LAMDA, 300 FOR BARCELONA
              
       union_of_query_labels{l,:} = or(targets(queryIndex(1,l), :), targets(queryIndex(2,l), : ));       
       absolute_union_of_query_labels{l,:} = nnz(union_of_query_labels{l,:} );     
       
       q_1 = data(queryIndex1,:);         % q1 & q2 are query pairs in the loop
       q_2 = data(queryIndex2,:);
       q1_rep{l,:} = repmat(q_1(l,:),N,1); % Make query matrix size to the same as data matrix size
       q2_rep{l,:} = repmat(q_2(l,:),N,1);      
       xor_data_q1new{l,:} = xor(data, q1_rep{l,:}); % xor of data and query matrices
       xor_data_q2new{l,:} = xor(data, q2_rep{l,:});       
       hamming_dist1{l,:} = sum(xor_data_q1new{l,:},2); % sum up rows to get hamming distances
       hamming_dist2{l,:} = sum(xor_data_q2new{l,:},2);
       
         
       
       X = zeros(2,N);
       X(1,:) = hamming_dist1{l,:};
       X(2,:) = hamming_dist2{l,:};
    
       X = (X)';   % pareto space for DMQIR & EMQR
       input = unique(X, 'rows');
       [K,~] = size(unique(X,'rows')); % The number of PPs = K
       pf_idx(l,:) = pareto_fronts(X, maxFront);

      
       %%%%%%%%%%%%%  Creation of Pareto Points for EMR %%%%%%%%%%%%%%%%%%       
           
        y1{l,:} = zeros(N,1);
        y1{l,:}(queryIndex1(l)) = 1; % Ranking  of query1 = 1
        y2{l,:} = zeros(N,1);
        y2{l,:}(queryIndex2(l)) = 1; % Ranking  of query2 = 1
    
        opts.p=100; 	% the number of landmarks picked 
        opts.r=5;  	% the number of nearest landmarks for representation (default 5)
        opts.a=0.99; 	% weight in manifold ranking, score = (I - aS)^(-1)y, default  0.99
        kmMaxIter = 5;
        kmNumRep = 1;
        
        [H A landmarks Z] = EMRcomputeModel(data); % Compute EMR
        simEMR1{l,:} = EMRscore(H ,A, y1{l,:});  % EMR Ranking
        simEMR2{l,:} = EMRscore(H ,A, y2{l,:});  % EMR Ranking
        dist1{l,:}   = 1-simEMR1{l,:};      % Dissimilarity 
        dist2{l,:}   = 1-simEMR2{l,:};      % Dissimilarity 
      
           
        X_EMR = zeros(2,N);
        X_EMR(1,:) = dist1{l,:};
        X_EMR(2,:) = dist2{l,:};
    
        X_EMR = (X_EMR)';  % pareto space for EMR
        input_EMR = unique(X_EMR, 'rows');
        [K_emr,~] = size(unique(X_EMR,'rows')); % The number of PPs = K
        pf_idx_EMR(l,:) = pareto_fronts(X_EMR, maxFront);
  
                 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   EMQR   %%%%%%%%%%%%%%%%%%%%%%   
   
       q1 = X(queryIndex1,:); 
       q2 = X(queryIndex2,:);     
       Cmn(l,:) = (q1(l,:) + q2(l,:)).'/2; % OPTIMAL PONT LOCATION
       
       Dissim{l,:} = EuDist2(Cmn(l,:), input);
       Dissim{l,:} = (Dissim{l,:})';

       [DissimSorted , DissimIndex ] = sort(Dissim{l,:}(:),'ascend'); % short & get indexes(before shorting)
 

       %%%%%%%%%%%%% Choose First P Shortest Pareto point to OPTIMAL POINT %%%%%%%%%%%%%%%%%%%%%%%%       
       
       DissimIndex_P                      =    DissimIndex(1:P, :); 
       Retrieved_PP_indexes{l,:}          =    ismember(X, input(DissimIndex_P,:),'rows');  % IF MQUR == 1
       Retrieved_Items_EMQR{l,:}          =    find(Retrieved_PP_indexes{l,:} ); 
             
       
       predicted_labels_EMQR{l,:} = targets(Retrieved_Items_EMQR{l,:} , :);        
       Retrieved_Items_EMQR{l,:}( all(~Retrieved_Items_EMQR{l,:},2), : ) = [];       
       
       
       diff_EMQR{l,:}     = ismember( predicted_labels_EMQR{l,:}, union_of_query_labels{l,:}  , 'rows');  
       if isempty( diff_EMQR{l,:})
            diff_EMQR{l,:} = 0;
       end        
       num_nz_EMQR(l,:)   = nnz( diff_EMQR{l,:}(:,1) ); % THE NUMBER OF MQUR == 1 POINTS
       s_EMQR{l,:}        = size(diff_EMQR{l,:}(:,1), 1); 
       
       acc_EMQR(l,:) = num_nz_EMQR(l,:) / s_EMQR{l,:};   % ACCURACY FOR OF RETRIVED ITEMS
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  acc of each front in maxFront for DMQIR & EMR   %%%%%%%%%%%%% 
   
    for j= 1:maxFront        
               
           
        Retrieved_Items_DMQIR{l,j}   =      pf_idx{l,j}(:,3); 
        predicted_labels_DMQIR{l,j}  =      targets(Retrieved_Items_DMQIR{l,j} , :); 
        
        Retrieved_Items_EMR{l,j}   =      pf_idx_EMR{l,j}(:,3); 
        predicted_labels_EMR{l,j}  =      targets(Retrieved_Items_EMR{l,j} , :); 
        
               
        diff_DMQIR{l,j}   = ismember( predicted_labels_DMQIR{l,j}, union_of_query_labels{l,:}  , 'rows'); % IF MQUR == 1
        if isempty( diff_DMQIR{l,j})
            diff_DMQIR{l,j} = 0;
        end
        num_nz_DMQIR(l,j) = nnz( diff_DMQIR{l,j}(:,1) );   % THE NUMBER OF MQUR == 1 POINTS
        s_DMQIR{l,j}      = size(diff_DMQIR{l,j}(:,1), 1); % THE NUMBER OF RETRIVED ITEMS 
       
        acc_DMQIR(l,j) = num_nz_DMQIR(l,j) / s_DMQIR{l,j}; % ACCURACY OF THE CURRENT FRONT
       
            
        
      
        
       Retrieved_Items_EMR{l,j}( all(~Retrieved_Items_EMR{l,j},2), : ) = [];       
         
        
       
       diff_EMR{l,j}     = ismember( predicted_labels_EMR{l,j}, union_of_query_labels{l,:}  , 'rows'); 
       if isempty( diff_EMR{l,j})
            diff_EMR{l,j} = 0;
       end        
       num_nz_EMR(l,j)   = nnz( diff_EMR{l,j}(:,1) );    % THE NUMBER OF MQUR == 1 POINTS
       s_EMR{l,j}        = size(diff_EMR{l,j}(:,1), 1);  % THE NUMBER OF RETRIVED ITEMS IN THE CURRENT FRONT
       
       acc_EMR(l,j) = num_nz_EMR(l,j) / s_EMR{l,j};      % ACCURACY OF THE CURRENT FRONT
      
       

        
        
    end  
    
    
    % Mean accuracy of the maxFront fronts     
    mean_acc_DMQIR(l,:) = sum(acc_DMQIR(l,:)) / maxFront;
    mean_acc_EMR(l,:)   = sum(acc_EMR(l,:)) / maxFront;
    
     
    
       
 end


 
 avg_acc_DMQIR = sum(mean_acc_DMQIR(:,1)) / l;
 avg_acc_EMR   = sum(mean_acc_EMR(:,1)) / l;
 
 avg_acc_EMIR  = mean(acc_EMQR);


