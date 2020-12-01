% Find opt. point and, then find distences between opt point and all pareto
% points by EMR ranking thats all
close all;
clear all;
clc;


%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/hashCodes_16.mat');
data = hashCodes_16;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/features/features_16.mat');
features = features_16;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
%}

%{
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/hashCodes_128.mat');
data = hashCodes_128;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/features/features_128.mat');
features = features_128;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
%}



load('Lamda/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
data = max(data,0);
load('Lamda/features/features_64.mat');
features = features_64;
load('Lamda/hashCodes/targets.mat');
targets = targets;
load('Lamda/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
%}

%{
load('Barcelona/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
data = max(data,0);
load('Barcelona/features/features_64.mat');
features = features_64;
load('Barcelona/hashCodes/targets.mat');
targets = targets;
load('Barcelona/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);
%}


% 431;827;1626
queryIndex1 = 599;
queryIndex2 = 600;
queryIndex3 = 1636;
queryIndex4 = 1637;
queryIndex5 = 1638;

q_1 = data(queryIndex1,:); 
q_2 = data(queryIndex2,:); 
q_3 = data(queryIndex3,:); 
q_4 = data(queryIndex4,:); 
q_5 = data(queryIndex5,:); 


N = length(filenames); 
q1new = repmat(q_1,N,1);
q2new = repmat(q_2,N,1);
q3new = repmat(q_3,N,1);
q4new = repmat(q_4,N,1);
q5new = repmat(q_5,N,1);

dist_1 = xor(data, q1new);
dist_2 = xor(data, q2new);
dist_3 = xor(data, q3new);
dist_4 = xor(data, q4new);
dist_5 = xor(data, q5new);


hamming_dist1 = sum(dist_1,2);
hamming_dist2 = sum(dist_2,2);
hamming_dist3 = sum(dist_3,2);
hamming_dist4 = sum(dist_4,2);

n_hamming_dist1 = mat2gray(hamming_dist1);
n_hamming_dist2 = mat2gray(hamming_dist2);
n_hamming_dist3 = mat2gray(hamming_dist3);
n_hamming_dist4 = mat2gray(hamming_dist4); 
 
X = zeros(4,N);
X(1,:) = hamming_dist1;
X(2,:) = hamming_dist2;
X(3,:) = hamming_dist3;
X(4,:) = hamming_dist4;

X = (X)';
input = unique(X, 'rows');

%hold off; scatter3(X(:,1),X(:,2),X(:,3),'k.'); hold on;
%hold off; scatter3(X(:,1),X(:,2),X(:,3),'o','MarkerFaceColor',[0 1 0])    
%set(gca,'FontSize',30); hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
union_of_query_labels = or(targets(queryIndex1, :), targets(queryIndex2, : ));
union_of_query_labels = or(union_of_query_labels  , targets(queryIndex3, : ));
union_of_query_labels = or(union_of_query_labels  , targets(queryIndex4, : ));
absolute_union_of_query_labels = nnz(union_of_query_labels);   
    
 for e = 1:N        
            MQUR_ALL(e,:) =  nnz( and(targets(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels ;
            
 end

MQUR_ONE = find(MQUR_ALL == 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%************* Cmn:(Optimum Point)*********

q1 = X(queryIndex1,:); 
q2 = X(queryIndex2,:); 
q3 = X(queryIndex3,:); 
q4 = X(queryIndex4,:);

Cmn = (q1(:) + q2(:) + q3(:) + q4(:)).'/2;
%plot3(q1 , q2, q3, 'p' , 'MarkerSize',20,'MarkerFaceColor',[1 0 1]);
%plot3(Cmn(1), Cmn(2), Cmn(3), 'd', 'MarkerSize',20 , 'MarkerFaceColor',[1 0 0]);
%plot3(q1 , q2, q3, 'gd' , 'LineWidth', 20); 
%plot3(Cmn(1), Cmn(2), Cmn(3), 'ro' , 'LineWidth', 10);

Dissim = EuDist2(Cmn, input, 'euclid');
Dissim = Dissim';
[DissimSorted, DissimIndex] = sort(Dissim,'ascend'); % short & get indexes(before shorting)


%%%%%%%%%%%%% Choose First P Shortest distances %%%%%%%%%%%%%%%%%%%%%%%%
P = 20;
DissimIndex_P          =    DissimIndex(1:P, :); 
Retrieved_PP_indexes   =    ismember(X, input(DissimIndex_P,:),'rows'); 
Retrieved_Items        =    find(Retrieved_PP_indexes); 
%{
scatter3(X(Retrieved_Items,1),X(Retrieved_Items,2),X(Retrieved_Items,3),'ys', 'LineWidth', 2); 
[x,y,z] = sphere;
radius = DissimSorted(P);
x = x * radius;
y = y * radius;
z = z * radius;
h = surfl(x + Cmn(1),y + Cmn(2),z + Cmn(3));
set(h, 'FaceAlpha', 0.1)
shading interp
%}
%%%%%%%%%%%%%%%%%%%%  ReRanking, rearrange P items by features  %%%%%%%%%%% 

% Add queries to Feature Pareto space, for creating Cmn_f
Retrieved_Items(end+1,:) = queryIndex1;
Retrieved_Items(end+1,:) = queryIndex2;
Retrieved_Items(end+1,:) = queryIndex3;
Retrieved_Items(end+1,:) = queryIndex4;
Retrieved_Items(end+1,:) = queryIndex5;

rtr_idx2_features = features(Retrieved_Items(:,1), :);

f1 = features(queryIndex1,:); 
f2 = features(queryIndex2,:);
f3 = features(queryIndex3,:);
f4 = features(queryIndex4,:);
f5 = features(queryIndex5,:);

dist_f1 = pdist2(f1 , rtr_idx2_features , 'euclid' );
dist_f2 = pdist2(f2 , rtr_idx2_features , 'euclid' );
dist_f3 = pdist2(f3 , rtr_idx2_features , 'euclid' );
dist_f4 = pdist2(f4 , rtr_idx2_features , 'euclid' );
dist_f5 = pdist2(f5 , rtr_idx2_features , 'euclid' );

[M2,B] = size(rtr_idx2_features(:,1));
YY = zeros(5,M2);
YY(1,:) = dist_f1;
YY(2,:) = dist_f2;
YY(3,:) = dist_f3;
YY(4,:) = dist_f4;
YY(5,:) = dist_f5;
YY = (YY)';


qf1 = YY(end-4,:); 
qf2 = YY(end-3,:); 
qf3 = YY(end-2,:);
qf4 = YY(end-1,:);
qf5 = YY(end,:);

Cmn_f = (qf1(:) + qf2(:) + qf3(:) + qf4(:) + qf5(:)).'/2;

Dissim_f                = EuDist2(YY, Cmn_f, 'euclid'); % Rank by EMR then Rerank by Euclidian ! ?
Dissim_ff               = [Retrieved_Items, Dissim_f];
DissimSorted_f          = sortrows(Dissim_ff, 2);

Retrieved_Items_Ranked  = DissimSorted_f(:,1);

% Now remove last three rows (qf1&qf2&qf3) from Retrieved items
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];


%%%%%%%%%%%%%%%%%%%%%%% Metrics & MQUR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted_labels  = targets(Retrieved_Items_Ranked,:);

[num_R, ~]  = size(Retrieved_Items_Ranked);     
 for e = 1:num_R         
     MQUR_Ranked(e,:) =  nnz( and(predicted_labels(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels;       
                        
 end
      
for ff = 1:num_R
    if  MQUR_Ranked(ff,:) ~= 1   %%%%%% MQUR !DEN FARKLI OLANLARI SEÇ 
        Retrieved_Items_Ranked(ff,:) = 0;   
        MQUR_Ranked(ff,:) = 0;
    end
end

Retrieved_Items_Ranked( all(~Retrieved_Items_Ranked,2), : ) = [];
MQUR_Ranked( all(~MQUR_Ranked,2), : ) = [];

 
 
predicted_labels_ranked  = targets(Retrieved_Items_Ranked,:);
%diff = ismember( predicted_labels_ranked, union_of_query_labels , 'rows' );
if isempty(MQUR_Ranked)
   MQUR_Ranked = 0;
end
  
num_nz = nnz( MQUR_Ranked(:,1) );
s = size(MQUR_Ranked(:,1), 1);
    
for j=1:s;        
    %Cummulative sum of the true-positive elements
    CUMM = cumsum(MQUR_Ranked);          
    Precision_AT_K(j,1) = ( CUMM(j,1)  ) ./ j;              
    Recall_AT_K(j,1) = ( CUMM(j,1)  ) ./ (num_nz); %              
end

avg_Precision = sum(Precision_AT_K(:,1) .* MQUR_Ranked(:,1) ) / num_nz;
avg_Precision(isnan(avg_Precision))=0;
% avg_Precision_OLD = sum(Precision_AT_K(:,1) ) / s;
 acc = num_nz / s;   % accuracy of the best cluster 


%{
plot(Recall_AT_K, Precision_AT_K);
hold off;
x = linspace(0,s);
plot( Precision_AT_K )
ylabel('Precision@k' ,'FontSize', 12)
xlabel('Number of Rterieved Items' ,'FontSize', 12) 
hold on;
%}

