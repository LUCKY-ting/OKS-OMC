
MATLAB Code for the paper: online kernel selection for online multi-label classification£¨submitted to Pattern Recognition journal£©

## guideline for how to use this code £¨requiring matlab r2016b or higher version£©

1. "run_OKS_OMC_GK.m" is a demo for our proposed accelerated OKS-OMC using P Gaussian kernels. 

2. "run_OKS_OMC_IK.m" is a demo for our proposed accelerated OKS-OMC using mS-3 isolation kernels. 
   "run_OKS_OMC_IK.m" relies on  "OKS_OMC_diffStep_checkConvergence.c" to run.
    You should first input "mex -largeArrayDims OKS_OMC_diffStep_checkConvergence.c" in the command window of matlab in order to build a executable mex-file.
    Then run "run_OKS_OMC_IK.m".

3. "run_OKS_OMC_GK_fixed.m" is a demo for the variant of our non-accelerated OKS-OMC that uses beta = 0 and P Gaussian kernels. 

4. "run_OKS_OMC_IK_fixed.m" is a demo for the variant of our non-accelerated OKS-OMC that uses beta = 0 and mS-3 isolation kernels.


ATTN: 
- This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Tingting ZHAI (zhtt@yzu.edu.cn).
- This package was developed by Tingting ZHAI (zhtt@yzu.edu.cn). For any problem concerning the code, please feel free to contact Mrs.ZHAI.