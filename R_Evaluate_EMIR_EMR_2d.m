
close all;
clear all;
clc;

%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/features/features_64.mat');
features = features_64;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
queryIndex = xlsread('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/qLabels_V2.xls');  % Reads randomly choosen query pairs from excell filequeryIndex = transpose( queryIndex ); 
%}

%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/hashCodes_16.mat');
data = hashCodes_16;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/features/features_16.mat');
features = features_16;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
queryIndex = xlsread('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/streets_2d.xls');  % Reads randomly choosen query pairs from excell filequeryIndex = transpose( queryIndex ); 
%}

%{
load('Lamda/hashCodes/hashCodes_16.mat');
data = hashCodes_16;
data = data > 0;
load('Lamda/features/features_16.mat');
features = features_16;
load('Lamda/hashCodes/targets.mat');
targets = targets;
load('Lamda/hashCodes/filenames.mat');
queryIndex = xlsread('Lamda/qGroups_2d.xls');
filenames = filenames;
N = length(filenames);
%}


load('Barcelona/hashCodes/hashCodes_16.mat');
data = hashCodes_16;
data = data > 0;
load('Barcelona/features/features_16.mat');
features = features_16;
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

for u=1:100
 for l = 1:240 % Number of Query Pairs , CAN TRY FOR DIFFERENT HAMMING RADIUS ALSO ?????? how?
              
       q_1 = data(queryIndex1,:);         % q1 & q2 are query pairs in the loop
       q_2 = data(queryIndex2,:);
       q1_rep{l,:} = repmat(q_1(l,:),N,1); % Make query matrix size to the same as data matrix size
       q2_rep{l,:} = repmat(q_2(l,:),N,1);      
       xor_data_q1new{l,:} = xor(data, q1_rep{l,:}); % xor of data and query matrices
       xor_data_q2new{l,:} = xor(data, q2_rep{l,:});       
       hamming_dist1{l,:} = sum(xor_data_q1new{l,:},2); % sum up rows to get hamming distances
       hamming_dist2{l,:} = sum(xor_data_q2new{l,:},2);
       %norm_hamming_dist1{l,:} =  hamming_dist1{l,:} / max( hamming_dist1{l,:}(:) ); % Normalize hamming  distances between 0&1
       %norm_hamming_dist2{l,:} =  hamming_dist2{l,:} / max( hamming_dist2{l,:}(:) );
       %dist1{l,:} = mat2gray(dist1{l,:}); % Normalize hamming  distances between 0&1
       %dist2{l,:} = mat2gray(dist2{l,:});     
        
       X{l,:} = zeros(2,N);
       X{l,:}(1,:) = hamming_dist1{l,:};
       X{l,:}(2,:) = hamming_dist2{l,:};    
       X{l,:} = (X{l,:})';    
       input{l,:} = unique(X{l,:}, 'rows');
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       union_of_query_labels{l,:} = or(targets(queryIndex(1,l), :), targets(queryIndex(2,l), : ));             
       absolute_union_of_query_labels{l,:} = nnz(union_of_query_labels{l,:} );
       
               
       %{   
        for e = 1:N        
            MQUR_ALL{l,:}(e,:) =  nnz( and(targets(e,:) , union_of_query_labels{l,:} ) ) / absolute_union_of_query_labels{l,:} ;
            
        end

        MQUR_ONE{l,:} = find(MQUR_ALL{l,:} == 1);
       %}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
     %************* Cmn: Optimum Point *********
        % queries location 
        q1 = X{l,:}(queryIndex1,:); 
        q2 = X{l,:}(queryIndex2,:);     
        Cmn(l,:) = (q1(l,:) + q2(l,:)).'/2; 
 
        %%%%%%%%%%%%%%%%%%%%%%%%%%% Add Cmn in the Pareto Space %%%%%%%%%%%%%
        input{l,:}(end+1,:) =  Cmn(l,:);     
        nSmp = size( input{l,:},1); 
        y1{l,:} = zeros(nSmp, 1); 
        y1{l,:}(end,:) = 1; %  Set The Rank of Cmn (Query) as 1 
 
 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EMR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    opts.p=50; 	% the number of landmarks picked (default 1000)
    opts.r=5;  	% the number of nearest landmarks for representation (default 5)
    opts.a=0.99; 	% weight in manifold ranking, score = (I - aS)^(-1)y, default  0.99
    kmMaxIter = 5;
    kmNumRep = 1;

 
    [dump,landmarks]=litekmeans(input{l,:} ,opts.p,'MaxIter',kmMaxIter,'Replicates',kmNumRep); 

    D = EuDist2( input{l,:} ,landmarks);
    dump = zeros(nSmp,opts.r); % dump was a 549x1 vector, but after this it is 549x2 vector !
    idx = dump;

    for i = 1:opts.r
        [dump(:,i),idx(:,i)] = min(D,[],2);
        temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
     D(temp) = 1e100;
    end

    dump = bsxfun(@rdivide,dump,dump(:,opts.r));
    dump = 0.75 * (1 - dump.^2);
    Gsdx = dump;
    Gidx = repmat([1:nSmp]',1,opts.r);
    Gjdx = idx;
    Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,opts.p);

    model.Z = Z';		%********************************************


    % Efficient Ranking
    feaSum = full(sum(Z,1));
    D = Z*feaSum';
    D = max(D, 1e-12);
    D = D.^(-.5);
    H = spdiags(D,0,nSmp,nSmp)*Z;

    C = speye(opts.p);
    A = H'*H-(1/opts.a)*C;

    % yo construction , for queries its 1 for rest its 0, should get index of the each query!
    % below we choose as 1 and three !  manually !!

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% EMR Ranking & Retrieval by Hash Codes %%%%%%%%%%%%%%%%%%%%%
    simEMR{l,:}                       = EMRscore(H ,A, y1{l,:}); 
    Dissim{l,:}                       = 1-simEMR{l,:} ;                 % Dissimilarity
    [DissimSorted, DissimIndex]       = sort(Dissim{l,:} ,'ascend');
    %input{l,:}(end,:)                = [];                           % Remove Cmn from space
    DissimIndex                       = DissimIndex(2:end, :);        % Remove Cmn's Index


    %%%%%%%%%%%%% Choose First P Shortest distances %%%%%%%%%%%%%%%%%%%%%%%%
    P = 5;
    DissimIndex_P                      =    DissimIndex(1:P, :); 
    Retrieved_PP_indexes{l,:}          =    ismember(X{l,:}, input{l,:}(DissimIndex_P,:),'rows'); 
    Retrieved_Items{l,:}               =    find(Retrieved_PP_indexes{l,:} );

       
    %%%%%%%%%%%%%%%%%%%  ReRanking, rearrange P items by features  %%%%%%%%%%% 
    
    % Add queries to Feature Pareto space, for creating Cmn_f
    Retrieved_Items{l,:}(end+1,:) = queryIndex1(:,l);
    Retrieved_Items{l,:}(end+1,:) = queryIndex2(:,l);

    rtr_idx2_features = features(Retrieved_Items{l,:}, :);
 
    % features of query pairs
    f1 = features(queryIndex1,:); 
    f2 = features(queryIndex2,:); 
 
    % Distance from each query pair features to retrieved items 
    dist_f1{l,:} = pdist2(f1(l,:) , rtr_idx2_features , 'euclid' );
    dist_f2{l,:} = pdist2(f2(l,:) , rtr_idx2_features , 'euclid' ); 
 
    % How many rows of trr_idx2
    [ M(l,:), ~] = size(rtr_idx2_features);

    % Create  2xM zero vector for assigne each distance (dis_f1 and f2) to them
    % YY is Pareto space formed by cnn features of retrived itmes,
    % which are retrieved by has codes
    YY{l,:} = zeros(2,M(l,:)); 
    YY{l,:}(1,:) = dist_f1{l,:};
    YY{l,:}(2,:) = dist_f2{l,:};
    YY{l,:} = (YY{l,:})';
 
    qf1(l,:) = YY{l,:}(end-1,:); 
    qf2(l,:) = YY{l,:}(end,:);
    
    % Find distance between each query for query pairs
    df(l,:) = pdist2(qf1(l,:),qf2(l,:),'euclid' );
    % Optimum Point of Pareto space formed by features of the retrived
    % items
    Cmn_f(l,:) = [df(l,:)/2 , df(l,:)/2];
    
    % DD is the distance between each retrieved items to optimum point
    Dissim_f{l,:}                = EuDist2(YY{l,:}, Cmn_f(l,:), 'euclid');
    Dissim_ff{l,:}               = [Retrieved_Items{l,:}, Dissim_f{l,:}];
    DissimSorted_f{l,:}          = sortrows(Dissim_ff{l,:}, 2);

    Retrieved_Items_Ranked{l,:}  = DissimSorted_f{l,:}(:,1);

    % Now remove last three rows (qf1&qf2&qf3) from Retrieved items
    Retrieved_Items_Ranked{l,:}(end) = [];
    Retrieved_Items_Ranked{l,:}(end) = [];
    predicted_labels{l,:} = targets(Retrieved_Items_Ranked{l,:} , :);      
    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       [num_R(l,:), ~]  = size(Retrieved_Items_Ranked{l,:} );     
        for e = 1:num_R(l,:)         
            MQUR_Ranked{l,:}(e,:) =  nnz( and(predicted_labels{l,:}(e,:) , union_of_query_labels{l,:} ) ) / absolute_union_of_query_labels{l,:} ;
            %MQUR_Ranked{l,:}(e,:) =  nnz(predicted_labels{l,:}(e,:) ) / absolute_union_of_query_labels{l,:}; % MQUR böylece 1 den büyük te olabilir.!      
        end
      
        for ff = 1:num_R(l,:)
           if  MQUR_Ranked{l,:}(ff,:) ~= 1  
               Retrieved_Items_Ranked{l,:}(ff,:) = 0; 
               MQUR_Ranked{l,:}(ff,:) = 0;                              
           end
        end
        
       Retrieved_Items_Ranked{l,:}( all(~Retrieved_Items_Ranked{l,:},2), : ) = [];       
       MQUR_Ranked{l,:}( all(~MQUR_Ranked{l,:},2), : ) = [];                                
       
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      predicted_labels_ranked{l,:} = targets(Retrieved_Items_Ranked{l,:} , :);   
      % diff{l,:} = ismember(predicted_labels_ranked{l,:}, union_of_query_labels{l,:}, 'rows'); 
       if isempty( MQUR_Ranked{l,:})
             MQUR_Ranked{l,:} = 0;
       end
      num_nz(l,:) = nnz( MQUR_Ranked{l,:}(:,1) );
      s{l,:} = size(MQUR_Ranked{l,:}(:,1), 1);      
    
       
       for j=1:s{l,:};
        
            % Cummulative sum of the true-positive elements
            CUMM{l,:} = cumsum(MQUR_Ranked{l,:});          
            Precision_AT_K{l,:}(j,1) = ( CUMM{l,:}(j,1)  ) / j;              
            Recall_AT_K{l,:}{j,1} = ( CUMM{l,:}(j,1)  ) / (num_nz(l,:)); %?????????????                    
       
       end  
    
    %acc(l,:) = num_nz(l,:) / s{l,:};   % accuracy of the best cluster 
    %avg_Precision(l,:) = sum(Precision_AT_K{l,:}(:,1) ) / s{l,:};
    avg_Precision(l,:) = sum(Precision_AT_K{l,:}(:,1)  .* MQUR_Ranked{l,:}(:,1) ) / num_nz(l,:);    
    avg_Precision(isnan(avg_Precision))=0;
    
 end
 
mAP(u) = sum(avg_Precision(:,1)) / l;
clear Precision_AT_K;

end

MAP = mean(mAP);


