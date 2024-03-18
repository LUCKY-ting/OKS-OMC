% FALT algorithm using a Gaussian kernel k(x,y)=exp(-||x-y||^2/2*scale)
% where the optimal scale is chosen from 2.^{-6:1:6} using only the initial
% data

clc
clear
dname = 'scene';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname '-test.mat']);
data = [trainData; testData];
labels = [trainLabel; testLabel];
clear  trainData testData trainLabel testLabel

times = 20;  % run 20 times for calculating mean accuracy
[n,d] = size(data);
L = size(labels,2);
initNum = round(n*0.2);

%hyperparameters for FALT
scale = 2.^(2);
eta = 2.^(-1);
maxIterNum = L;

sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);

macro_F1_score = zeros(times,1);
micro_F1_score = zeros(times,1);
hammingLoss = zeros(times,1);
rankingLoss = zeros(times,1);
subsetAccuracy = zeros(times,1);
oneError = zeros(times,1);
precision = zeros(times,1);
recall = zeros(times,1);
F1score = zeros(times,1);

tStart = tic;
for run = 1:times
    coff = zeros(n, L+1);
    SVsIdx = zeros(1, n);
    SVsNum = 0;
    tp = zeros(1,L);
    fp = zeros(1,L);
    fn = zeros(1,L);
    index = randperm(n);
    
    %initialization using the first part of data
    %%the kernel matrix was precalculated using only the initial data for accelerating the initialization
    initialData = data(index(1:initNum),:);
    kernelMatrix = rbfkernel(initialData', initialData',scale);
    
    %initialize the model using the first part of data
    for i = 1:n
        j = index(i);
        x = data(j,:)';
        y = labels(j,:);
        Y_t_size = nnz(y);
        pred_v = zeros(1,L+1);
        km = [];
        if i > 1
            if i<= initNum
                km = kernelMatrix(SVsIdx(1:SVsNum),i);
            else
                SVsMatrix = data(SVsIdx(1:SVsNum),:);
                km = rbfkernel(SVsMatrix',x,scale);
            end
            pred_v = km'* coff(1:SVsNum,:);
            pred_y = pred_v(1:L) > pred_v(L+1); %online prediction
        end
      
        if i > initNum
            %%%%%%update Online Metrics%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if nnz(pred_y) ~= 0
                precision(run) = precision(run) + (pred_y * y') / nnz(pred_y);
            elseif nnz(y) == 0
                precision(run) = precision(run) + 1;
            end
            
            if nnz(y) ~= 0
                recall(run) = recall(run) + (pred_y * y') / nnz(y);
            elseif nnz(pred_y) == 0
                recall(run) = recall(run) + 1;
            end
            
            hammingLoss(run) = hammingLoss(run) + nnz(pred_y + y == 1)/L ;
            if pred_y == y
                subsetAccuracy(run) = subsetAccuracy(run) + 1;
            end
            
            rele = find(y);
            irrele = find(~y);
            if ~isempty(rele) && ~isempty(irrele)
                misorder = 0;
                for kk = 1:size(rele,2)
                    misorder = misorder + nnz(pred_v(rele(kk)) <= pred_v(irrele));
                end
                rankingLoss(run) = rankingLoss(run) + misorder/(size(rele,2)*size(irrele,2));
                
                [~,id] = max(pred_v(1:L));
                if y(id) == 0
                    oneError(run) = oneError(run) + 1;
                end
            end
            
            for kk = 1:L
                if y(kk) == 1 && pred_y(kk) == 1
                    tp(kk) = tp(kk) + 1;
                elseif y(kk) == 1 && pred_y(kk) == 0
                    fn(kk) = fn(kk) + 1;
                elseif y(kk) == 0 && pred_y(kk) == 1
                    fp(kk) = fp(kk) + 1;
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        for iter = 1:maxIterNum
            a_t = 0;
            b_t = 0;
            cur_coeff = zeros(1, L+1);
            for k = 1:L
                if y(k) == 1 &&  (pred_v(k) -  pred_v(L+1) < 1)
                    a_t = a_t + 1;
                    cur_coeff(k) = eta / Y_t_size;
                    cur_coeff(L+1) = cur_coeff(L+1) - eta/Y_t_size;
                    pred_v(k) = pred_v(k) + cur_coeff(k);
                elseif y(k) == 0 &&  (pred_v(L+1) -  pred_v(k) < 1)
                    b_t = b_t + 1;
                    cur_coeff(k) = - eta / (L - Y_t_size);
                    cur_coeff(L+1) = cur_coeff(L+1) + eta/(L - Y_t_size);
                    pred_v(k) = pred_v(k) + cur_coeff(k);
                end
            end
            %re-compute the predicted value for all labels
            pred_v(L+1) =  pred_v(L+1) + cur_coeff(L+1);
            
            if a_t == 0 && b_t == 0
                break;
            end
            if iter == 1
                SVsNum = SVsNum + 1;
                if i<= initNum
                    SVsIdx(SVsNum) = i;
                else
                    SVsIdx(SVsNum) = j;
                end
                curId = SVsNum;
                km = [km; 1];
            end
            coff(curId, :) = coff(curId, :) + cur_coeff;
        end
        
        if i == initNum
            clear kernelMatrix
            SVsIdx(1:SVsNum) = index(SVsIdx(1:SVsNum));
        end
    end
  
    t = n - initNum; % the number of examples evaluated
    precision(run) = precision(run) / t;
    recall(run) = recall(run) / t;
    F1score(run) = 2 * precision(run) * recall(run) / (precision(run) + recall(run));
    hammingLoss(run) = hammingLoss(run)/t;
    subsetAccuracy(run) = subsetAccuracy(run)/t;
    rankingLoss(run) = rankingLoss(run)/t;
    oneError(run) = oneError(run)/t;
    
    macro_F1_score(run) = 0;
    for kk = 1:L
        this_F = 0;
        if tp(kk) ~= 0 || fp(kk) ~= 0 || fn(kk) ~= 0
            this_F = (2*tp(kk)) / (2*tp(kk) + fp(kk) + fn(kk));
        end
        macro_F1_score(run) = macro_F1_score(run) + this_F;
    end
    macro_F1_score(run) = macro_F1_score(run) / L;
    micro_F1_score(run) = (2*sum(tp)) / (2*sum(tp) + sum(fp) + sum(fn));
end
totalTime = toc(tStart);
avgTime = totalTime/times;


%-------------output result to file----------------------------------------
fid = fopen('scene-FALT-GK.csv','a');
fprintf(fid,'runTimes, maxIter, eta, scale, precision, +std, recall, +std, F1score, +std, macro_F1score, +std, micro_F1score, +std, hammingloss, +std, subsetAccuracy, +std, rankingLoss, +std, oneErr, +std, time [s] \n ');
fprintf(fid, '%d, %d, %g, %g, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n', ...
    times, maxIterNum, eta, scale, mean(precision), std(precision), mean(recall), std(recall), mean(F1score), std(F1score),...
    mean(macro_F1_score), std(macro_F1_score), mean(micro_F1_score), std(micro_F1_score), ...
    mean(hammingLoss), std(hammingLoss), mean(subsetAccuracy), std(subsetAccuracy),...
    mean(rankingLoss), std(rankingLoss), mean(oneError), std(oneError),avgTime);
fclose(fid);
metrics = [precision recall F1score macro_F1_score micro_F1_score hammingLoss subsetAccuracy rankingLoss oneError];
save('scene-FALT-GK-metrics.mat','metrics');


