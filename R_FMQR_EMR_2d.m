close all;
clear all;
clc;


load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/hashCodes_128.mat');
data = hashCodes_128;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/features/features_128.mat');
features = features_128;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/targets.mat');
targets = targets;
load('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/streetsDataset/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);



queryIndex1 = 1;
queryIndex2 = 472;


q_1 = data(queryIndex1,:); 
q_2 = data(queryIndex2,:); 
%q3 = data(queryIndex3,:); 

N = length(filenames); 
q1new = repmat(q_1,N,1);
q2new = repmat(q_2,N,1);
%q3new = repmat(q3,N,1);

dist_1 = xor(data, q1new);
dist_2 = xor(data, q2new);
%dist_3 = xor(data, q2new);

hamming_dist1 = sum(dist_1,2);
hamming_dist2 = sum(dist_2,2);
%hamming_dist3 = sum(dist_3,2);

n_hamming_dist1 = mat2gray(hamming_dist1);
n_hamming_dist2 = mat2gray(hamming_dist2);
%n_hamming_dist3 = mat2gray(hamming_dist3);
 
 
X = zeros(2,N);
X(1,:) = n_hamming_dist1;
X(2,:) = n_hamming_dist2;
%X(3,:) = n_hamming_dist3;
X = (X)';
input = unique(X, 'rows');
hold off; plot(X(:,1),X(:,2),'o','MarkerFaceColor',[0 1 0]); hold on;
set(gca,'FontSize',30); hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
union_of_query_labels = or(targets(queryIndex1, :), targets(queryIndex2, : ));
absolute_union_of_query_labels = nnz(union_of_query_labels);   
%{    
 for e = 1:N        
            MQUR_ALL(e,:) =  nnz( and(targets(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels ;
            
 end

MQUR_ONE = find(MQUR_ALL == 1);
scatter3(X(MQUR_ONE,1),X(MQUR_ONE,2),X(MQUR_ONE,3),'gs');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%************* Cmn:(Optimum Point)*********
q1 = X(queryIndex1,:); 
q2 = X(queryIndex2,:);
Cmn = (q1(:) + q2(:)).'/2;

% https://www.mathworks.com/matlabcentral/answers/44943-properties-of-the-plot-marker-shapes
plot(q1 , q2, 'p' , 'MarkerSize',20,'MarkerFaceColor',[1 0 1]);
plot(Cmn(1), Cmn(2), 'd', 'MarkerSize',20 , 'MarkerFaceColor',[1 0 0]);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Add Cmn in the Pareto Space and set its rank as 1 %%%%%%%%%%%%%%%
input(end+1,:) =  Cmn;
nSmp = size(input,1);
y1 = zeros(nSmp, 1);
y1(end) = 1; %  Set The Rank of Cmn (Query) as 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EMR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.p=100; 	% the number of landmarks picked (default 1000)
opts.r=5;  	% the number of nearest landmarks for representation (default 5)
opts.a=0.99; 	% weight in manifold ranking, score = (I - aS)^(-1)y, default  0.99
kmMaxIter = 5;
kmNumRep = 1;

 
[dump,landmarks]=litekmeans(input ,opts.p,'MaxIter',kmMaxIter,'Replicates',kmNumRep); 

D = EuDist2( input ,landmarks);
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

model.Z = Z';		

% Efficient Ranking
feaSum = full(sum(Z,1));
D = Z*feaSum';
D = max(D, 1e-12);
D = D.^(-.5);
H = spdiags(D,0,nSmp,nSmp)*Z;

C = speye(opts.p);
A = H'*H-(1/opts.a)*C;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% EMR Ranking & Retrieval by Hash Codes %%%%%%%
simEMR                      = EMRscore(H ,A, y1);
Dissim                      = 1-simEMR;                           % Dissimilarity
[DissimSorted, DissimIndex] = sort(Dissim,'ascend');
%input(end,:) =  [];                          % Remove Cmn from space
DissimIndex                 = DissimIndex(2:end, :);         % Remove Cmn's Index


%%%%%%%%%%%%% Choose First P Shortest distances %%%%%%%%%%%%%%%%%%%%%%%%
P = 30;
DissimIndex_P         =    DissimIndex(1:P, :); 
Retrieved_PP_indexes  =    ismember(X, input(DissimIndex_P,:),'rows'); 
Retrieved_Items       =    find(Retrieved_PP_indexes); 

plot(X(Retrieved_Items,1),X(Retrieved_Items,2),'ys', 'LineWidth', 2); 

%{
r=DissimSorted(P)/P;
x0=Cmn(1);
y0=Cmn(2);
theta = linspace(0,2*pi,100);
h = plot(x0 + r*cos(theta),y0 + r*sin(theta),'g', 'MarkerSize',5);
axis equal
%syms x y
%fimplicit((x-x0).^2 + (y-y0).^2 -r^2)
%axis equal
%}

%%%%%%%%%%%%%%%%%%%%  Re-Ranking, rearrange P items by features  %%%%%%%%%%% 

% Add queries to Feature Pareto space, for creating Cmn_f
Retrieved_Items(end+1,:) = queryIndex1;
Retrieved_Items(end+1,:) = queryIndex2;


%%%%%%%%%%%%%%%%%%%%  ReRanking, rearrange P items by features  %%%%%%%%%%% 
rtr_idx2_features = features(Retrieved_Items(:,1), :);

f1 = features(queryIndex1,:); 
f2 = features(queryIndex2,:);

dist_f1 = pdist2(f1 , rtr_idx2_features , 'euclid' );
dist_f2 = pdist2(f2 , rtr_idx2_features , 'euclid' );

[M2,B] = size(rtr_idx2_features(:,1));
YY = zeros(2,M2);
YY(1,:) = dist_f1;
YY(2,:) = dist_f2;
YY = (YY)';

qf1 = YY(end-1,:); 
qf2 = YY(end,:);

Cmn_f = (qf1(:) + qf2(:)).'/2;

Dissim_f                = EuDist2(YY, Cmn_f, 'euclid');
Dissim_ff               = [Retrieved_Items, Dissim_f];
DissimSorted_f          = sortrows(Dissim_ff, 2);

Retrieved_Items_Ranked  = DissimSorted_f(:,1);

% Now remove last two rows (qf1&qf2) from Retrieved items
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];

%%%%%%%%%%%%%%%%%%%%%%% Metrics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted_labels  = targets(Retrieved_Items_Ranked,:); % When doing Reranking

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
plot(X(Retrieved_Items_Ranked,1),X(Retrieved_Items_Ranked,2),'bs','LineWidth', 2);
 
 
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
