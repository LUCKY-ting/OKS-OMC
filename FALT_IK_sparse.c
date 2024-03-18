#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"


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

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 6) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "6 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0 && mxIsSparse(prhs[1])==0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "data/label matrix is not sparse!");
    }
    
    
    double *data, *labels, *w, *index, *x, *y, eta, a_t, b_t, *metrics, gamma_t;
    int i,j,k,p,q,N,d,L,low,high,nonzerosNum,low1,high1,epoch,o,Y_t_size,n_Y_t_size,iter,maxIterNum;
    mwIndex *ir, *jc, *ir1, *jc1;
    int *idx, maxId, t, start;
    double pred_count, true_count, diffNum, misorder, maxPred_v, sum_tp, sum_fp, sum_fn, this_F;
    
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    labels = mxGetPr(prhs[1]);
    index = mxGetPr(prhs[2]);
    epoch = mxGetScalar(prhs[3]);
    eta = mxGetScalar(prhs[4]);
    maxIterNum = mxGetScalar(prhs[5]);
    
    // a column is an instance
    d = (int)mxGetM(prhs[0]); //get Number of rows in array
    N = (int)mxGetN(prhs[0]); //get Number of columns in array
    L = (int)mxGetM(prhs[1]); //the dimension of each label vector
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    ir1 = mxGetIr(prhs[1]);
    jc1 = mxGetJc(prhs[1]);
    
    /* preparing outputs */
    plhs[0] = mxCreateDoubleMatrix(d, L+1, mxREAL);
    w = mxGetPr(plhs[0]);
    //store metric values: precision,recall,F1,macro_F1_score,micro_F1_score,hammingLoss,subsetAccuracy,rankingLoss,oneError
    plhs[1] = mxCreateDoubleMatrix(1, 9, mxREAL);
    metrics = mxGetPr(plhs[1]);
    start = (int)round(N*0.2);
    
    double *pred_v = Malloc(double,L+1);
    double *pred_y = Malloc(double,L);  //store the predicted result
    double *tp = Malloc(double,L);
    double *fp = Malloc(double,L);
    double *fn = Malloc(double,L);
    for (k = 0; k < L; k++){
        tp[k] = 0;
        fp[k] = 0;
        fn[k] = 0;
    }
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
            
            // compute each predicted value
            for (k = 0; k <= L; k++){
                pred_v[k] = 0;
                for (p = 0; p < nonzerosNum; p++){
                    pred_v[k] += w[k*d + idx[p]] * x[p];
                }
            }
            
            //---------------- update Online Metrics----------------------
            if (t > start){ //the data after start is used to evaluate the model
                pred_count = 0;
                true_count = 0;
                maxPred_v = -DBL_MAX;
                maxId = -1;
                for (k = 0; k < L; k++){
                    if (pred_v[k] > pred_v[L]){
                        pred_y[k] = 1;
                        pred_count = pred_count + 1;
                        true_count = true_count + pred_y[k]*y[k];
                    }else
                        pred_y[k] = 0;
                    
                    if (pred_v[k] > maxPred_v){
                        maxPred_v = pred_v[k];
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
                    if (y[k] == 1 &&  pred_y[k] == 1)
                        tp[k] = tp[k] + 1;
                    else if (y[k] == 1 && pred_y[k] == 0)
                        fn[k] = fn[k] + 1;
                    else if (y[k] == 0 && pred_y[k] == 1)
                        fp[k] = fp[k] + 1;
                    
                    if (y[k] != pred_y[k])
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
                            if (pred_v[releIdx[k]] <= pred_v[irreIdx[q]])
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
            
            //update each multi-label model
            for (iter = 0; iter < maxIterNum; iter++){
                a_t = 0;
                b_t = 0;
                gamma_t = 0;
                for (k = 0; k < L; k++){  // update the predictive models
                    if (y[k] == 1 && pred_v[k] - pred_v[L] < 1){
                        a_t++;
                        for (p = 0; p < nonzerosNum; p++){
                            w[k*d + idx[p]] = w[k*d + idx[p]] + eta/Y_t_size * x[p];
                            w[L*d + idx[p]] = w[L*d + idx[p]] - eta/Y_t_size * x[p];
                        }
                        pred_v[k] = pred_v[k] + eta/Y_t_size;
                        gamma_t = gamma_t + 1.0/Y_t_size;
                    }else if (y[k] == 0 && pred_v[L] - pred_v[k] < 1){
                        b_t++;
                        for (p = 0; p < nonzerosNum; p++){
                            w[k*d + idx[p]] = w[k*d + idx[p]] - eta/n_Y_t_size * x[p];
                            w[L*d + idx[p]] = w[L*d + idx[p]] + eta/n_Y_t_size * x[p];
                        }
                        pred_v[k] = pred_v[k] - eta/n_Y_t_size;
                        gamma_t = gamma_t - 1.0/n_Y_t_size;
                    }
                }
                pred_v[L] = pred_v[L] - eta*gamma_t;
                if (a_t == 0 && b_t == 0)    break;
              
            }
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
    
    free(pred_v);
    free(pred_y);
    free(tp);
    free(fp);
    free(fn);
}

