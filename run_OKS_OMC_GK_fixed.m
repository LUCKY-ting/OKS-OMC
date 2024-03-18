%A variant of our non-accelerated OKS-OMC. It uses 13 Gaussian kernels of the form k(x,y)=exp(-||x-y||^2/2*scale)
% where scale takes in the set {2.^{-6:1:6}}. It adopts beta = 0.

clc
clear
dname = 'scene';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname '-test.mat']);
data = [trainData; testData];
labels = [trainLabel; testLabel];

times = 20;  % run 20 times for calculating mean accuracy

[n,d] = size(data);
L = size(labels,2);
initNum = round(n*0.2);

%hyperparameters for OKS-OMC
maxIterNum = L;
lambda = 10^(-6);
eta = 2.^(-1);
beta = 0;
scale = 2.^(-6:1:6); %RBF kernel hyperparameter
P = size(scale,2); % the size of the predefined kernel set

%for the reproducibility of experimental results
sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);

%%for storing the performance metrics' values
macro_F1_score = zeros(times,1);
micro_F1_score = zeros(times,1);
hammingLoss = zeros(times,1);
rankingLoss = zeros(times,1);
subsetAccuracy = zeros(times,1);
oneError = zeros(times,1);
precision = zeros(times,1);
recall = zeros(times,1);
F1score = zeros(times,1);
output_miu = zeros(times,P);

tStart = tic;
for run = 1:times
    coff = zeros(n, L+1, P);
    SVsIdx = zeros(1, n, P);
    SVsNum = zeros(1,P);
    bar_miu = ones(1,P);
    miu = bar_miu /sum(bar_miu);
    wSNorm = zeros(1,L+1,P);
    km = cell(P,1);
    tp = zeros(1,L);
    fp = zeros(1,L);
    fn = zeros(1,L);
    
    index = randperm(n);
    kernelMatrix = zeros(initNum,initNum,P);
    initialData = data(index(1:initNum),:);
    for p = 1:P   %%the kernel matrix was precalculated using only the initial data for accelerating the computing
        kernelMatrix(:,:,p) = rbfkernel(initialData', initialData',scale(p));
    end
    
    for i = 1:n
        j = index(i);
        x = data(j,:)';
        y = labels(j,:);
        Y_t_size = nnz(y);
        
        pred_v = zeros(1,L+1,P);
        pred_val = zeros(1,L+1);
        if i > 1
            %make a combined prediction
            for p = 1:P
                if i<= initNum
                    km{p} = kernelMatrix(SVsIdx(1,1:SVsNum(p),p),i,p);
                else
                    SVsMatrix = data(SVsIdx(1,1:SVsNum(p),p),:);
                    km{p} = rbfkernel(SVsMatrix',x,scale(p));
                end
                pred_v(:,:,p) = km{p}'* coff(1:SVsNum(p),:,p);
                pred_val = pred_val + miu(p) * pred_v(:,:,p);
            end
        end
        pred_y = pred_val(1:L) > pred_val(L+1);
        
        %%%%%%%%%%%%%%%---online metrics calculation--%%%%%%%%%%%%%%%%%%%%%
        if i > initNum %the data after initNum is used to evaluate the model
            %%% for computing precision, recall and F1score
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
            
            %%% for computing hamming loss and subsetaccuracy
            hammingLoss(run) = hammingLoss(run) + nnz(pred_y + y == 1)/L ;
            if pred_y == y
                subsetAccuracy(run) = subsetAccuracy(run) + 1;
            end
            
            %%% for computing ranking loss and one-error
            rele = find(y);
            irrele = find(~y);
            if ~isempty(rele) && ~isempty(irrele)
                misorder = 0;
                for kk = 1:size(rele,2)
                    misorder = misorder + nnz(pred_val(rele(kk)) <= pred_val(irrele));
                end
                rankingLoss(run) = rankingLoss(run) + misorder/(size(rele,2)*size(irrele,2));
                
                [~,id] = max(pred_val(1:L));
                if y(id) == 0
                    oneError(run) = oneError(run) + 1;
                end
            end
            
            %%% for computing MacroF1 and MicroF1
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
        
        %perform online kernel selection and online model updating
        for p = 1:P %update each multi-label model in each mapped feature space
            for iter = 1:maxIterNum
                regu = lambda/2 * sum(wSNorm(:,:,p),2);
                coff(:,:,p) = (1 - lambda*eta) * coff(:,:,p);
                wSNorm(:,:,p) = (1 - lambda*eta)^2 * wSNorm(:,:,p);
                a_t = 0;
                b_t = 0;
                lloss = 0;
                cur_coeff = zeros(1, L+1);
                for k = 1:L
                    if y(k) == 1 &&  (pred_v(1,k,p) -  pred_v(1,L+1,p) < 1)
                        a_t = a_t + 1;
                        lloss = lloss + (1 - (pred_v(1,k,p) - pred_v(1,L+1,p)))/Y_t_size;
                        wSNorm(1,k,p) = wSNorm(1,k,p) + (eta/Y_t_size)^2 + 2*eta*(1 - lambda*eta)/Y_t_size * pred_v(1,k,p);
                        cur_coeff(k) = eta / Y_t_size;
                        cur_coeff(L+1) = cur_coeff(L+1) - eta/Y_t_size;
                        pred_v(1,k,p) = (1 - lambda*eta) * pred_v(1,k,p) + cur_coeff(k);  %re-compute the predicted value
                    elseif y(k) == 0 &&  (pred_v(1,L+1,p) -  pred_v(1,k,p) < 1)
                        b_t = b_t + 1;
                        lloss = lloss + (1 - (pred_v(1,L+1,p) - pred_v(1,k,p)))/(L - Y_t_size);
                        wSNorm(1,k,p) = wSNorm(1,k,p) + (eta/(L-Y_t_size))^2 - 2*eta*(1 - lambda*eta)/(L - Y_t_size)*pred_v(1,k,p);
                        cur_coeff(k) = - eta / (L - Y_t_size);
                        cur_coeff(L+1) = cur_coeff(L+1) + eta/(L - Y_t_size);
                        pred_v(1,k,p) = (1 - lambda*eta) * pred_v(1,k,p) + cur_coeff(k);  %re-compute the predicted value
                    else
                        pred_v(1,k,p) = (1 - lambda*eta) * pred_v(1,k,p);
                    end
                end
                
                %%% gamma_t * (-eta) =  cur_coeff(L+1)
                wSNorm(1,L+1,p) = wSNorm(1,L+1,p) + (cur_coeff(L+1))^2 + 2*(1 - lambda*eta)*cur_coeff(L+1)*pred_v(1,L+1,p);
                pred_v(1,L+1,p) = (1 - lambda*eta) *  pred_v(1,L+1,p) + cur_coeff(L+1);
                
                %update combination coefficients
                floss = lloss + regu;
                bar_miu(p) = bar_miu(p) * exp(-beta * floss);
                
                if a_t == 0 && b_t == 0
                    break;
                end
                
                if iter == 1 %store SVs and their coefficients
                    SVsNum(p) = SVsNum(p) + 1;
                    if i<= initNum
                        SVsIdx(1,SVsNum(p),p) = i;
                    else
                        SVsIdx(1,SVsNum(p),p) = j;
                    end
                    curId = SVsNum(p);
                    km{p} = [km{p}; 1];
                end
                coff(curId,:,p) = coff(curId, :, p) + cur_coeff;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        miu = bar_miu/sum(bar_miu);
        if sum(bar_miu) < 1e-10   %%%% to avoid dividing by zero
            bar_miu = bar_miu / max(bar_miu);
        end

        if i == initNum
            clear kernelMatrix
            for p = 1:P
                SVsIdx(1,1:SVsNum(p),p) = index(SVsIdx(1,1:SVsNum(p),p));
            end
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
    
    output_miu(run,:) = miu;
end

totalTime = toc(tStart);
avgTime = totalTime/times;


%-------------output result to file----------------------------------------
fid = fopen(['OKS-OMC-GK-fixed-' dname '.csv'],'a');
fprintf(fid,'runTimes, maxIter, P, lambda, eta, beta, precision, +std, recall, +std, F1score, +std, macro_F1score, +std, micro_F1score, +std, hammingloss, +std, subsetAccuracy, +std, rankingLoss, +std, oneErr, +std, time [s] \n ');
fprintf(fid, '%d, %d, %d, %g, %g, %g, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n', ...
    times, maxIterNum, P,lambda, eta, beta, mean(precision), std(precision), mean(recall), std(recall), mean(F1score), std(F1score),...
    mean(macro_F1_score), std(macro_F1_score), mean(micro_F1_score), std(micro_F1_score), ...
    mean(hammingLoss), std(hammingLoss), mean(subsetAccuracy), std(subsetAccuracy),...
    mean(rankingLoss), std(rankingLoss), mean(oneError), std(oneError),avgTime);
fclose(fid);

metrics = [precision recall F1score macro_F1_score micro_F1_score hammingLoss subsetAccuracy rankingLoss oneError];
save(['OKS-OMC-GK-fixed-metrics-' dname '.mat'],'metrics');
save(['OKS-OMC-GK-fixed-outputmiu-' dname '.mat'],'output_miu');




