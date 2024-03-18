% FALT algorithm using an isolation kernel 
% with psi chosen from 2.^{4:1:S}

clc
clear
dname = 'mediamill';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname '-test.mat']);
data = [trainData; testData];
labels = [trainLabel; testLabel];
clear  trainData testData trainLabel testLabel
[n,d] = size(data);
L = size(labels,2);

%%hyperparameters for creating an isolation kernel
% use the first 20% data in the stream to create isolation kernels
psi = 2.^(5); % the size of each sample
t = 2*d; % the number of sampling
initNum = round(0.2*n); 

%%running settings for FALT
epoch = 1;
times = 10;  % run 10 times for calculating mean metrics
maxIter = L;

%hyperparameters for FALT
eta = 2.^(-1);

sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);

metrics = zeros(times,9);
tStart = tic;
for run = 1:times
    index = randperm(n);
    %% create isolation kernels using 20% data in the stream and generate the mapped data
    mappingData = IKspace_mapping_gpu_v3(data(index(1:initNum),:), data, psi, t);
    mappedlabels = sparse(labels);
    [w, metrics(run,:)] = FALT_IK_sparse(mappingData',mappedlabels', index, epoch, eta, maxIter);
end
totalTime = toc(tStart);
avgTime = totalTime/times;

%---------------------output result to file----------------------------------------
fid = fopen(['FALT-IK-' dname '.csv'],'a');
fprintf(fid,'runTimes, epoch, maxIter, psi, t, eta, precision, +std, recall, +std, F1score, +std, macro_F1score, +std, micro_F1score, +std, hammingloss, +std, subsetAccuracy, +std, rankingLoss, +std, oneErr, +std, time [s] \n ');
fprintf(fid, '%d, %d, %d, %d, %d, %g, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n', ...
    times, epoch, maxIter, psi, t, eta, mean(metrics(:,1)), std(metrics(:,1)), mean(metrics(:,2)), std(metrics(:,2)), mean(metrics(:,3)), std(metrics(:,3)),...
    mean(metrics(:,4)), std(metrics(:,4)), mean(metrics(:,5)), std(metrics(:,5)), ...
    mean(metrics(:,6)), std(metrics(:,6)), mean(metrics(:,7)), std(metrics(:,7)),...
    mean(metrics(:,8)), std(metrics(:,8)), mean(metrics(:,9)), std(metrics(:,9)),avgTime);
fclose(fid);
save(['FALT-IK-' dname '-psi_' num2str(psi)  '.mat'],'metrics');


