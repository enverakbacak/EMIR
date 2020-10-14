% https://www.mathworks.com/matlabcentral/answers/361878-getting-the-data-points-of-each-cluster-in-kmeans
close all;
clear all;
clc;

load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/hashCodes_32.mat');
data = hashCodes_32;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/filenames.mat');
filenames = filenames;

N = length(filenames);

queryIndex = xlsread('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/streets_2d.xls');  % Reads randomly choosen query pairs from excell file
queryIndex = transpose( queryIndex ); 
queryIndex1 = queryIndex(1,:); % First element of Query Pair
queryIndex2 = queryIndex(2,:); % Second element of Qury Pair

       
maxFront = 10;  

for u =1:50

 for l = 1:240 % Number of Query Pairs 
    
      %%%%%%%%%%%%%  Creation of Pareto Points for EMR %%%%%%%%%%%%%%%%%%%%
        
           
        y1{l,:} = zeros(N,1);
        y1{l,:}(queryIndex1(l)) = 1; % Ranking  of query1 = 1
        y2{l,:} = zeros(N,1);
        y2{l,:}(queryIndex2(l)) = 1; % Ranking  of query2 = 1
    
    

        [H A landmarks Z] = EMRcomputeModel(data); % Compute EMR
        simEMR1{l,:} = EMRscore(H ,A, y1{l,:});  % EMR Ranking
        simEMR2{l,:} = EMRscore(H ,A, y2{l,:});  % EMR Ranking
        dist1{l,:}   = 1-simEMR1{l,:};      % Dissimilarity 
        dist2{l,:}   = 1-simEMR2{l,:};      % Dissimilarity 
      
           
        X_EMR = zeros(2,N);
        X_EMR(1,:) = dist1{l,:};
        X_EMR(2,:) = dist2{l,:};
    
        X_EMR = (X_EMR)';  
       
%%%%%%%%%%%%%%%%%%%%%%%%%% Pareto Fronts for EMR %%%%%%%%%%%%%%%%%%%%%%%%%%
          pf_idx(l,:) = pareto_fronts(X_EMR, maxFront);
          
          q1_label{l,:} = targets(queryIndex1(l), : ); % Label vector of Query 1
          q2_label{l,:} = targets(queryIndex2(l), : ); % Label vector of Query 2
        
          b{l,:} = or(q1_label{l,:} , q2_label{l,:});    % beta in the equation 7 
          absolute_b{l,:} = nnz(b{l,:});                 % Number of non-zero elements in the beta, nnz is a Matlab Func.
       
 
 
%%%%%%%%%%%%%%%%%%%% Precision & Recall for EMR  %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % NOW FIND Precision values of first 10 front for each query pairs
    for j= 1:maxFront
        predicted_labels_EMR{l,j} =      targets(pf_idx{l,j}(:,3),:);
        union_of_query_labels_EMR{l,:} = or(targets(queryIndex(1,l), :), targets(queryIndex(2,l), : ));
        absolute_union_of_query_labels_EMR{l,:}(j,:) = nnz(union_of_query_labels_EMR{l,:} ); 
        
        diff_EMR{l,j} = ismember( predicted_labels_EMR{l,j}, union_of_query_labels_EMR{l,:}  , 'rows'); 
        num_nz_EMR(l,j) = nnz( diff_EMR{l,j}(:,1) );
        s_EMR{l,j} = size(diff_EMR{l,j}(:,1), 1);
        
        for jj=1:s_EMR{l,j};
        
            % Cummulative sum of the true-positive elements
            CUMM_EMR{l,j} = cumsum(diff_EMR{l,j});          
            Precision_AT_K_EMR{l,j}(jj,1) = ( CUMM_EMR{l,j}(jj,1)  ) / jj;              
            Recall_AT_K_EMR{l,j}{jj,1} = ( CUMM_EMR{l,j}(jj,1)  ) / (num_nz_EMR(l,j)); %?????????????                    
       
        end  
        
        acc_EMR(l,j) = num_nz_EMR(l,j) / s_EMR{l,j};   % accuracy of the best cluster 
        %avg_Precision_EMR(l,j) = sum(Precision_AT_K_EMR{l,j}(:,1) ) / s_EMR{l,j};
        avg_Precision_EMR(l,j) = sum(Precision_AT_K_EMR{l,j}(:,1) .* diff_EMR{l,j} ) / num_nz_EMR(l,j);
        avg_Precision_EMR(isnan(avg_Precision_EMR))=0;

    end
    
        mean_avg_Precision_EMR(l,:) = sum(avg_Precision_EMR(l,:)) / maxFront;
        mean_acc_EMR(l,:) = sum(acc_EMR(l,:)) / maxFront;
       
 end
 

mAP_EMR(u) = sum(mean_avg_Precision_EMR(:,1)) / l;
clear Precision_AT_K_EMR

end

MAP = mean(mAP_EMR);

