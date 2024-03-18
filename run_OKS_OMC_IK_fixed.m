%A variant of our non-accelerated OKS-OMC. It uses mS-3 isolation kernels 
% where psi takes values in the set {2.^{4:1:mS}}. It adopts beta = 0.


clc
clear
dname = 'mediamill';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname  '-test.mat']);
data = [trainData; testData];
labels = [trainLabel; testLabel];
clear trainData testData trainLabel testLabel
[n,d] = size(data);
L = size(labels, 2);

%%hyperparameters for creating isolation kernels
initNum = round(0.2*n); % use the first 20% data in the stream to create isolation kernels
maxS = floor(log2(initNum/2));
mS = min(maxS, 13);
psi = 2.^(4:1:mS);
t = 2*d; % the number of sampling
P = size(psi,2);

%%running settings for OKS-OMC 
epoch = 1;
times = 10;
maxIter = L;

%%hyperparameters for OKS-OMC 
lambda = 10^(-8);
eta = 2^(-1);
beta = 0;

 
%% for the reproducibility of experimental results
sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);

%% for storing the performance metrics' values
metrics = zeros(times,9);
output_miu = zeros(times,P);
tStart = tic;
for run = 1:times
    index = randperm(n);
    %% create isolation kernels using 20% data in the stream and generate the mapped data
    mappingData = [];
    for p = 1:P
        mapped = IKspace_mapping_gpu_v3(data(index(1:initNum),:), data, psi(p), t);
        mappingData = [mappingData mapped];
    end
    dataLabels = sparse(labels);
%     save(fileName,'mappingData','labels');

    [v, a, mu, wsNorm, metrics(run,:)] = OKS_OMC_IK_fixed(mappingData',dataLabels', index, epoch, t, P, lambda, eta, beta, maxIter);
    output_miu(run,:) = mu;
end
totalTime = toc(tStart);
avgTime = totalTime /times;


%%------------------------output result to file-------------------------------------------------
fid = fopen(['OKS-OMC-IK-fixed-' dname '.csv'],'a');
fprintf(fid,'runTimes, epoch, maxIter, P, t, lambda, eta, beta, precision, +std, recall, +std, F1score, +std, macro_F1score, +std, micro_F1score, +std, hammingloss, +std, subsetAccuracy, +std, rankingLoss, +std, oneErr, +std, time [s] \n ');
fprintf(fid, '%d, %d, %d, %d, %d, %g, %g, %g, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n', ...
    times, epoch, maxIter, P,t, lambda, eta, beta, mean(metrics(:,1)), std(metrics(:,1)), mean(metrics(:,2)), std(metrics(:,2)), mean(metrics(:,3)), std(metrics(:,3)),...
    mean(metrics(:,4)), std(metrics(:,4)), mean(metrics(:,5)), std(metrics(:,5)), ...
    mean(metrics(:,6)), std(metrics(:,6)), mean(metrics(:,7)), std(metrics(:,7)),...
    mean(metrics(:,8)), std(metrics(:,8)), mean(metrics(:,9)), std(metrics(:,9)),avgTime);
fclose(fid);
save(['OKS-OMC-IK-fixed-metrics-' dname '.mat'],'metrics');
save(['OKS-OMC-IK-fixed-outputmiu-' dname '.mat'],'output_miu');



