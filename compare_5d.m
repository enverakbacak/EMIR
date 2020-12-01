
% https://www.mathworks.com/matlabcentral/answers/361878-getting-the-data-points-of-each-cluster-in-kmeans
close all;
clear all;
clc;

%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/hashCodes_32.mat');
data = hashCodes_32;
%load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/features/features_64.mat');
%features = features_64;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
queryIndex = xlsread('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/qLabels_5d.xls');  % Reads randomly choosen query pairs from excell filequeryIndex = transpose( queryIndex ); 
%}


%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
%load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/features/features_16.mat');
%features = features_16;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/filenames.mat');
queryIndex = xlsread('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/streets_5d.xls');  % Reads randomly choosen query pairs from excell file
filenames = filenames;
N = length(filenames);
%}

%{
load('Lamda/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
data = max(data,0);
%load('Lamda/features/features_16.mat');
%features = features_16;
load('Lamda/hashCodes/targets.mat');
targets = targets;
load('Lamda/hashCodes/filenames.mat');
queryIndex = xlsread('Lamda/qGroups_5d.xls');
filenames = filenames;
N = length(filenames);
%}


load('Barcelona/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
data = max(data,0);
%load('Barcelona/features/features_16.mat');
%features = features_16;
load('Barcelona/hashCodes/targets.mat');
targets = targets;
load('Barcelona/hashCodes/filenames.mat');
queryIndex = xlsread('Barcelona/qGroups_5d.xls');
filenames = filenames;
N = length(filenames);
%}


queryIndex = transpose( queryIndex ); 
queryIndex1 = queryIndex(1,:);        % First element of Query Triplets
queryIndex2 = queryIndex(2,:);        % Second element of Query Triplets
queryIndex3 = queryIndex(3,:); 
queryIndex4 = queryIndex(4,:); 
queryIndex5 = queryIndex(5,:);  

maxFront = 3;  

 for l = 1:240 % Number of Query Pairs , CAN TRY FOR DIFFERENT HAMMING RADIUS ALSO ?????? how?
       
       union_of_query_labels{l,:} = or(targets(queryIndex(1,l), :), targets(queryIndex(2,l), : )); 
       union_of_query_labels{l,:} = or(union_of_query_labels{l,:},  targets(queryIndex(3,l), : )); 
       union_of_query_labels{l,:} = or(union_of_query_labels{l,:},  targets(queryIndex(4,l), : ));
       union_of_query_labels{l,:} = or(union_of_query_labels{l,:},  targets(queryIndex(5,l), : ));
       absolute_union_of_query_labels{l,:} = nnz(union_of_query_labels{l,:} );
     
       q_1 = data(queryIndex1,:);         % q1 & q2 are query pairs in the loop
       q_2 = data(queryIndex2,:);
       q_3 = data(queryIndex3,:);
       q_4 = data(queryIndex4,:);
       q_5 = data(queryIndex5,:);
       
       q1_rep{l,:} = repmat(q_1(l,:),N,1); % Make query matrix size to the same as data matrix size
       q2_rep{l,:} = repmat(q_2(l,:),N,1);  
       q3_rep{l,:} = repmat(q_3(l,:),N,1); 
       q4_rep{l,:} = repmat(q_4(l,:),N,1); 
       q5_rep{l,:} = repmat(q_5(l,:),N,1);
       
       xor_data_q1new{l,:} = xor(data, q1_rep{l,:}); % xor of data and query matrices
       xor_data_q2new{l,:} = xor(data, q2_rep{l,:}); 
       xor_data_q3new{l,:} = xor(data, q3_rep{l,:});
       xor_data_q4new{l,:} = xor(data, q4_rep{l,:});
       xor_data_q5new{l,:} = xor(data, q5_rep{l,:});
       
       hamming_dist1{l,:} = sum(xor_data_q1new{l,:},2); % sum up rows to get hamming distances
       hamming_dist2{l,:} = sum(xor_data_q2new{l,:},2);
       hamming_dist3{l,:} = sum(xor_data_q3new{l,:},2);
       hamming_dist4{l,:} = sum(xor_data_q4new{l,:},2);
       hamming_dist5{l,:} = sum(xor_data_q5new{l,:},2);
             
       
       X = zeros(5,N);
       X(1,:) = hamming_dist1{l,:};
       X(2,:) = hamming_dist2{l,:};    
       X(3,:) = hamming_dist3{l,:}; 
       X(4,:) = hamming_dist4{l,:};
       X(5,:) = hamming_dist5{l,:};
       X = (X)';        
       input = unique(X, 'rows');
       pf_idx(l,:) = pareto_fronts(X, maxFront);

      %{
       %%%%%%%%%%%%%  Creation of Pareto Points for EMR %%%%%%%%%%%%%%%%%%       
           
        y1{l,:} = zeros(N,1);
        y1{l,:}(queryIndex1(l)) = 1; % Ranking  of query1 = 1
        y2{l,:} = zeros(N,1);
        y2{l,:}(queryIndex2(l)) = 1; % Ranking  of query2 = 1
        y3{l,:} = zeros(N,1);
        y3{l,:}(queryIndex3(l)) = 1; % Ranking  of query2 = 1
        y4{l,:} = zeros(N,1);
        y4{l,:}(queryIndex4(l)) = 1; % Ranking  of query2 = 1
        y5{l,:} = zeros(N,1);
        y5{l,:}(queryIndex5(l)) = 1; % Ranking  of query2 = 1
        
        [H A landmarks Z] = EMRcomputeModel(data); % Compute EMR
        simEMR1{l,:} = EMRscore(H ,A, y1{l,:});  % EMR Ranking
        simEMR2{l,:} = EMRscore(H ,A, y2{l,:});  % EMR Ranking
        simEMR3{l,:} = EMRscore(H ,A, y3{l,:});  % EMR Ranking
        simEMR4{l,:} = EMRscore(H ,A, y4{l,:});  % EMR Ranking
        simEMR5{l,:} = EMRscore(H ,A, y5{l,:});  % EMR Ranking
        
        dist1{l,:}   = 1-simEMR1{l,:};      % Dissimilarity 
        dist2{l,:}   = 1-simEMR2{l,:};      % Dissimilarity 
        dist3{l,:}   = 1-simEMR3{l,:};      % Dissimilarity 
        dist4{l,:}   = 1-simEMR4{l,:};      % Dissimilarity 
        dist5{l,:}   = 1-simEMR5{l,:};      % Dissimilarity 
           
        X_EMR = zeros(5,N);
        X_EMR(1,:) = dist1{l,:};
        X_EMR(2,:) = dist2{l,:};
        X_EMR(3,:) = dist3{l,:};
        X_EMR(4,:) = dist4{l,:};
        X_EMR(5,:) = dist5{l,:};
        X_EMR = (X_EMR)';  
        input_EMR = unique(X_EMR, 'rows');
        pf_idx_EMR(l,:) = pareto_fronts(X_EMR, maxFront);
  
      %}           
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   EMQR   %%%%%%%%%%%%%%%%%%%%%%   
   
       q1 = X(queryIndex1,:); 
       q2 = X(queryIndex2,:); 
       q3 = X(queryIndex3,:);
       q4 = X(queryIndex4,:);
       q5 = X(queryIndex5,:);
    
       Cmn(l,:) = (q1(l,:) + q2(l,:) + q3(l,:) + q4(l,:) + q5(l,:)).'/2;     
       
       Dissim{l,:} = EuDist2(Cmn(l,:), input);
       Dissim{l,:} = (Dissim{l,:})';

       [DissimSorted , DissimIndex ] = sort(Dissim{l,:}(:),'ascend'); % short & get indexes(before shorting)
 

       %%%%%%%%%%%%% Choose First P Shortest Pareto point to OPTIMAL POINT %%%%%%%%%%%%%%%%%%%%%%%%       
       P = 3;
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
               
           
        Retrieved_Items_DMQIR{l,j}   =      pf_idx{l,j}(:,6); 
        predicted_labels_DMQIR{l,j}  =      targets(Retrieved_Items_DMQIR{l,j} , :); 
        
        %Retrieved_Items_EMR{l,j}     =      pf_idx_EMR{l,j}(:,6); 
        %predicted_labels_EMR{l,j}    =      targets(Retrieved_Items_EMR{l,j} , :); 
        
               
        diff_DMQIR{l,j}   = ismember( predicted_labels_DMQIR{l,j}, union_of_query_labels{l,:}  , 'rows'); % IF MQUR == 1
        if isempty( diff_DMQIR{l,j})
            diff_DMQIR{l,j} = 0;
        end
        num_nz_DMQIR(l,j) = nnz( diff_DMQIR{l,j}(:,1) );   % THE NUMBER OF MQUR == 1 POINTS
        s_DMQIR{l,j}      = size(diff_DMQIR{l,j}(:,1), 1); % THE NUMBER OF RETRIVED ITEMS IN THE CURRENT FRONT
       
        acc_DMQIR(l,j) = num_nz_DMQIR(l,j) / s_DMQIR{l,j}; % ACCURACY OF THE CURRENT FRONT
       
         
        
       %{
       diff_EMR{l,j}     = ismember( predicted_labels_EMR{l,j}, union_of_query_labels{l,:}  , 'rows'); 
       if isempty( diff_EMR{l,j})
            diff_EMR{l,j} = 0;
       end        
       num_nz_EMR(l,j)   = nnz( diff_EMR{l,j}(:,1) );    % THE NUMBER OF MQUR == 1 POINTS
       s_EMR{l,j}        = size(diff_EMR{l,j}(:,1), 1);  % THE NUMBER OF RETRIVED ITEMS IN THE CURRENT FRONT
       
       acc_EMR(l,j) = num_nz_EMR(l,j) / s_EMR{l,j};      % ACCURACY OF THE CURRENT FRONT
      %}
       

        
        
    end  
    
    
    % Mean accuracy of the maxFront fronts     
    mean_acc_DMQIR(l,:) = sum(acc_DMQIR(l,:)) / maxFront;
    %mean_acc_EMR(l,:)   = sum(acc_EMR(l,:)) / maxFront;
    
     
    
       
 end


 
 avg_acc_DMQIR = sum(mean_acc_DMQIR(:,1)) / l;
 %avg_acc_EMR   = sum(mean_acc_EMR(:,1)) / l;
 
 avg_acc_EMQR  = mean(acc_EMQR);


