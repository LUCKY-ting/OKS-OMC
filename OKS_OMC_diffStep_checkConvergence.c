#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"
#include "string.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

double maxfun(double a, double b)
{
    if (a >= b) return a;
    else return b;
}

double minfun(double a, double b)
{
    if (a <= b) return a;
    else return b;
}

double uniform(double a, double b)
{
    return ((double) rand())/ RAND_MAX * (b -a) + a;
}

int binornd(double p)
{
    int x;
    double u;
    u = uniform(0.0, 1.0);
    x = (u <= p)? 1:0;
    return(x);
}


int getRandInt(int lowerLimit, int upperLimit){ // get an randomized interger in [lowerLimit, upperLimit]
    return lowerLimit + rand() % (upperLimit - lowerLimit + 1);
}

void randPerm(double *index, int N){
    int i, r1, r2, tmp;
    for(i=0; i < N; i++){
        r1 = getRandInt(0, N-1);
        r2 = getRandInt(0, N-1);
        if (r1!=r2){
            tmp =  index[r1];
            index[r1]= index[r2];
            index[r2] = tmp;
        }
    }
}

double squareNorm(double *x, int len){
    int i;
    double sum = 0;
    for(i = 0;i < len; i++){
        sum = sum + x[i] * x[i];
    }
    return sum;
}

int arr_maxvID(double *array, int len){
    int i,maxId = -1;
    double  maxv = -DBL_MAX;
    for (i=0; i<len; i++){
        if (array[i] > maxv){
            maxv = array[i];
            maxId = i;
        }
    }
    return maxId;    
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 10) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "10 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0 && mxIsSparse(prhs[1])==0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "data/label matrix is not sparse!");
    }
    if (mxGetScalar(prhs[8]) == 0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "require beta is not zero.");
    }
    
    double *data, *labels, *w, *mu, *a, *v, *wSNorm, *index, *x, *y, lambda, eta, beta, a_t, b_t,lloss,regu,floss, sum_bar_mu, gamma_t, *metrics;
    int i,j,k,p,q,N,d,L,low,high,nonzerosNum,low1,high1,epoch,o,Y_t_size,n_Y_t_size,iter,maxIterNum;
    mwIndex *ir, *jc, *ir1, *jc1;
    int *idx, P, sn, t, maxId, start, constantTimes, opt;
    double pred_count, true_count, diffNum, misorder, maxPred_v, sum_tp, sum_fp, sum_fn, this_F,diff;
    char * stage;
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    labels = mxGetPr(prhs[1]);
    index = mxGetPr(prhs[2]);
    epoch = mxGetScalar(prhs[3]);
    sn    = mxGetScalar(prhs[4]);  // the sampling times for the isolation kernel
    P     = mxGetScalar(prhs[5]);  // the number of base kernels
    lambda= mxGetScalar(prhs[6]);
    eta = mxGetScalar(prhs[7]);
    beta = mxGetScalar(prhs[8]);
    maxIterNum = mxGetScalar(prhs[9]);
    
    // a column is an instance
    d = (int)mxGetM(prhs[0]); //get Number of rows in array
    N = (int)mxGetN(prhs[0]); //get Number of columns in array
    L = (int)mxGetM(prhs[1]); //the dimension of each label vector
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    ir1 = mxGetIr(prhs[1]);
    jc1 = mxGetJc(prhs[1]);
    
    /* preparing outputs */ /*auxiliary variables for performing efficient updating */
    plhs[0] = mxCreateDoubleMatrix(d, L+1, mxREAL);
    v = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(P, L+1, mxREAL);
    a = mxGetPr(plhs[1]);  //represent each w = a*v
    plhs[2] = mxCreateDoubleMatrix(P, 1, mxREAL);
    mu = mxGetPr(plhs[2]);
    //wSNorm = mxGetPr(mxCreateDoubleMatrix(P, L+1, mxREAL)); //record the squared l2 norm of each w
    plhs[3] = mxCreateDoubleMatrix(P, L+1, mxREAL);
    wSNorm = mxGetPr(plhs[3]);
    //store the online metric values://precision,recall,F1,macro_F1_score,micro_F1_score,hammingLoss,subsetAccuracy,rankingLoss,oneError
    plhs[4] = mxCreateDoubleMatrix(1, 9, mxREAL);
    metrics = mxGetPr(plhs[4]);
    start = (int)round(N*0.2);
    
    for (k = 0; k < P*(L+1); k++){
        a[k] = 1;
    }
    
    double **pred_v = Malloc(double*,P);
    for (k = 0 ; k < P; k++)
        pred_v[k] = Malloc(double,L+1);
    
    double *mk_pred_v = Malloc(double,L+1); //store the multi-kernel predicted scores
    double *mk_pred_y = Malloc(double,L);  //store the multi-kernel predicted result
    
    double *bar_mu = Malloc(double,P);
    for (p = 0; p < P; p++){
        bar_mu[p] = 1;
        mu[p] = 1.0/P;
    }
    
    double *tp = Malloc(double,L);
    double *fp = Malloc(double,L);
    double *fn = Malloc(double,L);
    for (k = 0; k < L; k++){
        tp[k] = 0;
        fp[k] = 0;
        fn[k] = 0;
    }
    
    double *prev_mu = Malloc(double,P);
    constantTimes = 0;
    stage = "first";
    opt = -1;
    srand(0);
    
    for (o = 1; o <= epoch; o++){
        if (o > 1) randPerm(index, N);
        /* start loop */
        for(i = 0; i < N; i++)
        {
            j = index[i] - 1;
            t = (o - 1)*N + i + 1;
            
            // get each instance
            low = jc[j]; high = jc[j+1];
            nonzerosNum = high - low;
            x = Malloc(double,nonzerosNum);
            idx = Malloc(int,nonzerosNum); // the indices of the non-zero values in x
            for (k = low; k < high; k++){
                x[k-low] = data[k];
                idx[k-low] = ir[k];
            }
            
            // get each label vector
            y = Malloc(double,L);
            for (k = 0; k < L; k++){
                y[k] = 0;
            }
            low1 = jc1[j]; high1 = jc1[j+1];
            for (k = low1; k < high1; k++){
                y[ir1[k]] = labels[k];
            }
            Y_t_size = high1 - low1; // the number of relevant labels
            n_Y_t_size = L - Y_t_size; // the number of irrelevant labels
            memcpy(prev_mu, mu, sizeof(double)*P);
            
            if (strcmp(stage,"first")==0){
                //make a combined prediction
                for (k = 0; k <= L; k++){
                    mk_pred_v[k] = 0;
                    for(p = 0; p < P; p++){
                        pred_v[p][k] = 0;
                        for (q = sn*p; q < sn*(p+1); q++){
                            pred_v[p][k] += v[k*d + idx[q]] * x[q];
                        }
                        pred_v[p][k] = a[k*P+p] * pred_v[p][k];
                        mk_pred_v[k] += mu[p] * pred_v[p][k];
                    }
                }
            }else{//make a prediction using the optimal classifier
              
                for (k = 0; k <= L; k++){
                    mk_pred_v[k] = 0;
                    p = opt;
                    pred_v[p][k] = 0;
                    for (q = sn*p; q < sn*(p+1); q++){
                        pred_v[p][k] += v[k*d + idx[q]] * x[q];
                    }
                    pred_v[p][k] = a[k*P+p] * pred_v[p][k];
                    mk_pred_v[k] = pred_v[p][k];
                }
            }
          
           
            //---------------- update Online Metrics-------------------------------
            if (t > start){ //the data after start is used to evaluate the model
                pred_count = 0;
                true_count = 0;
                maxPred_v = -DBL_MAX;
                maxId = -1;
                for (k = 0; k < L; k++){
                    if (mk_pred_v[k] > mk_pred_v[L]){
                        mk_pred_y[k] = 1;
                        pred_count = pred_count + 1;
                        true_count = true_count + mk_pred_y[k]*y[k];
                    }else
                        mk_pred_y[k] = 0;
                    
                    if (mk_pred_v[k] > maxPred_v){
                        maxPred_v = mk_pred_v[k];
                        maxId = k;
                    }
                }
                
                if (pred_count != 0) //update precision
                    metrics[0] = metrics[0] + true_count / pred_count;
                else if (Y_t_size == 0)
                    metrics[0] = metrics[0] + 1;
                
                if (Y_t_size != 0) //update recall
                    metrics[1] = metrics[1] + true_count / Y_t_size;
                else if (pred_count == 0)
                    metrics[1] = metrics[1] + 1;
                
                diffNum = 0;
                int *releIdx = Malloc(int,Y_t_size);
                int *irreIdx = Malloc(int,n_Y_t_size);
                int rc = 0,irc = 0;
                for (k = 0; k < L; k++){  //update for macroF1 and microF1
                    if (y[k] == 1 &&  mk_pred_y[k] == 1)
                        tp[k] = tp[k] + 1;
                    else if (y[k] == 1 && mk_pred_y[k] == 0)
                        fn[k] = fn[k] + 1;
                    else if (y[k] == 0 && mk_pred_y[k] == 1)
                        fp[k] = fp[k] + 1;
                    
                    if (y[k] != mk_pred_y[k])
                        diffNum = diffNum + 1;
                    
                    if(y[k] == 1)
                        releIdx[rc++] = k;
                    else
                        irreIdx[irc++] = k;
                }
                
                metrics[5] = metrics[5] + diffNum / L; //update hamming loss
                metrics[6] = metrics[6] + (diffNum == 0); //update subset accuracy
                
                misorder = 0;
                if(Y_t_size !=0 && n_Y_t_size !=0){ //update ranking loss and one error
                    for (k = 0; k < Y_t_size; k++){
                        for (q = 0; q < n_Y_t_size; q++){
                            if (mk_pred_v[releIdx[k]] <= mk_pred_v[irreIdx[q]])
                                misorder = misorder + 1;
                        }
                    }
                    metrics[7] = metrics[7] + misorder / (Y_t_size*n_Y_t_size);
                    if (y[maxId] == 0)
                        metrics[8] = metrics[8] + 1;
                }
                free(releIdx);
                free(irreIdx);
            }
            //---------------- end of updating Online Metrics-------------------------------
            
            if (strcmp(stage,"first")==0){ //perform online kernel selection and online model updating
                for (p = 0; p < P; p++){ //update each multi-label model in each mapped feature space (w = a*v)
                    for (iter = 0; iter < maxIterNum; iter++){
                        a_t = 0;
                        b_t = 0;
                        lloss = 0;
                        gamma_t = 0;
                        regu = lambda/2 * wSNorm[L*P+p];  //compute ||w||^2 for (L+1)-th w
                        a[L*P+p] = (1 - lambda*eta) * a[L*P+p]; //update a for (L+1)-th w
                        wSNorm[L*P+p] = pow(1-lambda*eta, 2) * wSNorm[L*P+p]; //update ||w||^2 for (L+1)-th w
                        for (k = 0; k < L; k++){
                            regu += lambda/2 * wSNorm[k*P+p]; //compute ||w||^2 for k-th w
                            a[k*P+p] = (1 - lambda*eta) * a[k*P+p]; //update a for k-th w
                            wSNorm[k*P+p] = pow(1-lambda*eta, 2) * wSNorm[k*P+p]; //update ||w||^2 for k-th w
                            if (y[k] == 1 && pred_v[p][k] - pred_v[p][L] < 1){
                                a_t++;
                                lloss += (1 - (pred_v[p][k] - pred_v[p][L]))/Y_t_size;
                                wSNorm[k*P+p] += pow(eta/Y_t_size,2) + 2*(1-lambda*eta)*eta/Y_t_size * pred_v[p][k]; //update ||w||^2 for k-th w
                                
                                for (q = sn*p; q < sn*(p+1); q++){
                                    v[k*d + idx[q]] = v[k*d + idx[q]] + eta/Y_t_size/a[k*P+p] * x[q];//update v for k-th w
                                    v[L*d + idx[q]] = v[L*d + idx[q]] - eta/Y_t_size/a[L*P+p] * x[q]; //update v for (L+1)-th w
                                }
                                pred_v[p][k] = (1 - lambda*eta) * pred_v[p][k] + eta/Y_t_size;  //update the predicted score
                                gamma_t = gamma_t + 1.0/Y_t_size;
                            }else if (y[k]==0 && pred_v[p][L] - pred_v[p][k] < 1){
                                b_t++;
                                lloss += (1 - (pred_v[p][L] - pred_v[p][k]))/n_Y_t_size;
                                wSNorm[k*P+p] += pow(eta/n_Y_t_size,2) - 2*(1-lambda*eta)*eta/n_Y_t_size * pred_v[p][k]; //update ||w||^2
                                
                                for (q = sn*p; q < sn*(p+1); q++){
                                    v[k*d + idx[q]] = v[k*d + idx[q]] - eta/n_Y_t_size/a[k*P+p] * x[q];//update v for k-th w
                                    v[L*d + idx[q]] = v[L*d + idx[q]] + eta/n_Y_t_size/a[L*P+p] * x[q];//update v for (L+1)-th w
                                }
                                pred_v[p][k] = (1 - lambda*eta) * pred_v[p][k] - eta/n_Y_t_size;  //update the predicted score
                                gamma_t = gamma_t - 1.0/n_Y_t_size;
                            }else
                                pred_v[p][k] = (1 - lambda*eta) * pred_v[p][k];//update the predicted score
                        }
                        
                        wSNorm[L*P+p] += pow(eta*gamma_t,2) - 2*(1-lambda*eta)*eta*gamma_t*pred_v[p][L];
                        pred_v[p][L] = (1 - lambda*eta) * pred_v[p][L] - eta*gamma_t;
                       
                        //update bar_mu and mu
                        floss = lloss + regu;
                        bar_mu[p] = bar_mu[p] * exp(-beta * floss);
                        
                        if (a_t == 0 && b_t == 0)
                            break;
                    }
                }
                sum_bar_mu = 0;
                for (p = 0; p < P; p++){
                    sum_bar_mu += bar_mu[p];
                }
                if (sum_bar_mu < 1e-10){  // to avoid dividing by zero
                    for (p = 0; p < P; p++){
                        bar_mu[p] = bar_mu[p]*1e10;
                    }
                    sum_bar_mu = sum_bar_mu * 1e10;
                }
                diff = 0;
                for (p = 0; p < P; p++){
                    mu[p] = bar_mu[p]/sum_bar_mu; //update mu
                    diff = diff + fabs(mu[p] - prev_mu[p]);
                }
                //check convergence by observating the number of consecutive times that miu remains unchanged
                if (diff < 1e-5){
                      constantTimes ++;
                      if (constantTimes == 100){
                          stage = "second";  printf("current round: %d \n", t);
                          opt = arr_maxvID(mu, P);
                      }
                }else
                      constantTimes = 0;
                
            }else{ //only update the model corresponding to the optimal kernel
                p = opt;
                for (iter = 0; iter < maxIterNum; iter++){
                    a_t = 0;
                    b_t = 0;
                    gamma_t = 0;
                    a[L*P+p] = (1 - lambda*eta) * a[L*P+p]; //update a for (L+1)-th w
                    for (k = 0; k < L; k++){
                        a[k*P+p] = (1 - lambda*eta) * a[k*P+p]; //update a for k-th w
                        if (y[k] == 1 && pred_v[p][k] - pred_v[p][L] < 1){
                            a_t++;
                            for (q = sn*p; q < sn*(p+1); q++){
                                v[k*d + idx[q]] = v[k*d + idx[q]] + eta/Y_t_size/a[k*P+p] * x[q];//update v for k-th w
                                v[L*d + idx[q]] = v[L*d + idx[q]] - eta/Y_t_size/a[L*P+p] * x[q]; //update v for (L+1)-th w
                            }
                            pred_v[p][k] = (1 - lambda*eta) * pred_v[p][k] + eta/Y_t_size;  //update the predicted score
                            gamma_t = gamma_t + 1.0/Y_t_size;
                        }else if (y[k]==0 && pred_v[p][L] - pred_v[p][k] < 1){
                            b_t++;
                            for (q = sn*p; q < sn*(p+1); q++){
                                v[k*d + idx[q]] = v[k*d + idx[q]] - eta/n_Y_t_size/a[k*P+p] * x[q];//update v for k-th w
                                v[L*d + idx[q]] = v[L*d + idx[q]] + eta/n_Y_t_size/a[L*P+p] * x[q]; //update v for (L+1)-th w
                            }
                            pred_v[p][k] = (1 - lambda*eta) * pred_v[p][k] - eta/n_Y_t_size;  //update the predicted score
                            gamma_t = gamma_t - 1.0/n_Y_t_size;
                        }else
                            pred_v[p][k] = (1 - lambda*eta) * pred_v[p][k];//update the predicted score
                    }
                    pred_v[p][L] = (1 - lambda*eta) * pred_v[p][L] - eta*gamma_t;
                    if (a_t == 0 && b_t == 0)
                        break;
                }
            }
            //printf("sum_bar_mu =%e \n",sum_bar_mu);
            free(x);
            free(idx);
            free(y); 
        }
    }
    
    
    t = t - start; // the number of examples evaluated
    metrics[0] = metrics[0]/t;
    metrics[1] = metrics[1]/t;
    if (metrics[0] != 0 && metrics[1] != 0)
        metrics[2] = 2 * metrics[0] * metrics[1] / (metrics[0] + metrics[1]);
    else
        metrics[2] = 0;
    metrics[5] = metrics[5]/t;
    metrics[6] = metrics[6]/t;
    metrics[7] = metrics[7]/t;
    metrics[8] = metrics[8]/t;
    
    metrics[3] = 0;
    sum_tp = 0;
    sum_fp = 0;
    sum_fn = 0;
    for(k = 0; k < L; k++){
        this_F = 0;
        if (tp[k] != 0 || fp[k]!= 0 || fn[k] != 0)
            this_F = (2*tp[k]) / (2*tp[k] + fp[k] + fn[k]);
        metrics[3] = metrics[3] + this_F;
        
        sum_tp = sum_tp + tp[k];
        sum_fp = sum_fp + fp[k];
        sum_fn = sum_fn + fn[k];
    }
    metrics[3] = metrics[3] / L;
    metrics[4] = (2*sum_tp) / (2*sum_tp + sum_fp + sum_fn);
    
    
    for (k = 0 ; k < P; k++)
        free(pred_v[k]);
    free(pred_v);
    free(bar_mu);
    free(mk_pred_v);
    free(mk_pred_y);
    free(tp);
    free(fp);
    free(fn);
    //mxDestroyArray(wSNorm);
}
