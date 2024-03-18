function [ndata] = IKspace_mapping_gpu_v3 (Sdata, data, psi, t)
% mapping data to Isolation Kernel space using the partitions got from Sdata

[sn,~] = size(Sdata);
[n,~]=size(data);
nnzRowIdx = zeros(n, t); % store the row indice of each non-zero element
nnzColIdx = zeros(n, t); % store the column indice of each non-zero element

chunk = 1000; % chunk the data into smaller size and proceed the testing sequentially to prevent Matlab Out-of-Memory



for i = 1:t
    subIndex = datasample(1:sn, psi, 'Replace', false);
    partdata = Sdata(subIndex,:);
    X = gpuArray(partdata)';
    
    for k = 1:ceil(n/chunk)
        ind_start = 1+(k-1)*chunk;
        if k*chunk<n
            ind_end = k*chunk;
        else
            ind_end = n;
        end
        Y = gpuArray(data(ind_start:ind_end,:))';
        %%%%---a fast way to compute the paired distances--%%%%% each column of X and Y is an observation
        K = X'*Y;
        [~,Mx] = size(X);
        [~,My] = size(Y);
        
        K = 2*K;
        K = K - sum(X.^2,1)'*ones(1,My);
        K = K - ones(Mx,1)*sum(Y.^2,1);
        dis=-K;
        %%%%%%%%-----------%%%%%%%
        [~, centerIdx] = min(dis);
        centerIdx = gather(centerIdx);
        rowIdx = (i-1)*psi + centerIdx;
        colIdx = ind_start:ind_end;
        nnzRowIdx(ind_start:ind_end,i) = rowIdx';
        nnzColIdx(ind_start:ind_end,i) = colIdx';
    end
end

ndata = sparse(nnzRowIdx(:),nnzColIdx(:),ones(n*t,1),psi*t,n);
ndata = sqrt(1/t) * ndata';

end

