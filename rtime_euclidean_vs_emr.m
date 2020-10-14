close all;
clear all;

random = rand(500000,5);
w=5000;
MQOP_3 = random(w,:,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Euclidian
tic
d_euclidian = EuDist2(random, MQOP_3, 'euclidian');
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


nSmp = size(random,1);
y0 = zeros(nSmp,1);
y0(w) = 1; 

opts = [];
p = 1000;
opts.p = p;
r=5;
opts.r = r;
a = 0.99;
opts.a = a;
opts.mode = 'kmeans';
% Landmark selection
if strcmp(opts.mode,'kmeans')
    kmMaxIter = 5;
    if isfield(opts,'kmMaxIter')
        kmMaxIter = opts.kmMaxIter;
    end
    kmNumRep = 1;
    if isfield(opts,'kmNumRep')
        kmNumRep = opts.kmNumRep;
    end
    [dump,landmarks]=litekmeans(random,p,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
     clear kmMaxIter kmNumRep
elseif strcmp(mode,'random')
    indSmp = randperm(nSmp);
    landmarks = data(indSmp(1:p),:);
    clear indSmp
else
    error('mode does not support!');
end

model.landmarks = landmarks;
model.a = a;
model.r = r;

% Z construction
D = EuDist2(random,landmarks);
dump = zeros(nSmp,r);
idx = dump;
for i = 1:r
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    D(temp) = 1e100;
end
dump = bsxfun(@rdivide,dump,dump(:,r));
dump = 0.75 * (1 - dump.^2);
Gsdx = dump;
Gidx = repmat([1:nSmp]',1,r);
Gjdx = idx;
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);

model.Z = Z';

% Efficient Ranking
feaSum = full(sum(Z,1));
D = Z*feaSum';
D = max(D, 1e-12);
D = D.^(-.5);
H = spdiags(D,0,nSmp,nSmp)*Z;

C = speye(p);
A = H'*H-(1/a)*C;

tmp = H'*y0;
tmp = A\tmp;
score = y0 - H*tmp;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMRtest OUT OF SAMPLE 
tic
y = EMRtest(MQOP_3,model);
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EMR
tic
y1 = zeros(nSmp, 1);
y1(w) = 1; %  Set The Rank of Cmn (Query) as 1 
[score, model] = EMR(random,y1,opts);
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}









