#!/usr/bin/env python
import pandas as pd
import glob

print('Reading METRIC_GROUP_0/apex.0.csv')
df = pd.read_csv('METRIC_GROUP_0/apex.0.csv')
for file_name in glob.glob('METRIC_GROUP_[1-9]*/apex.0.csv'):
    print("Reading", file_name)
    x = pd.read_csv(file_name)
    for column in x:
        if column in df:
            continue
        print("Getting column", column)
        extracted = x[column]
        df = df.join(extracted)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 20)
pd.set_option('display.max_colwidth', 20)

# Compute total flops
df["FLOPS"] = ((64 * (df["rocm:::SQ_INSTS_VALU_ADD_F64:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_MUL_F64:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_FMA_F64:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_FMA_F64:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_TRANS_F64:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_MFMA_MOPS_F64:device=0"])) +
               (32 * (df["rocm:::SQ_INSTS_VALU_ADD_F32:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_MUL_F32:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_FMA_F32:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_FMA_F32:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_TRANS_F32:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_MFMA_MOPS_F32:device=0"])) +
               (16 * (df["rocm:::SQ_INSTS_VALU_ADD_F16:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_MUL_F16:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_FMA_F16:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_FMA_F16:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_TRANS_F16:device=0"] +
                      df["rocm:::SQ_INSTS_VALU_MFMA_MOPS_F16:device=0"])))/df["num samples/calls"]
df["IOPS"] = ((64 * (df["rocm:::SQ_INSTS_VALU_INT32:device=0"] +
                     df["rocm:::SQ_INSTS_VALU_INT64:device=0"]))/df["num samples/calls"])

# Compute LDS bytes transferred
df["LDS"] = ( 32 * 4 * ((df["rocm:::SQ_LDS_IDX_ACTIVE:device=0"] -
                         df["rocm:::SQ_LDS_BANK_CONFLICT:device=0"])/ df["num samples/calls"]))
# Compute vL1D bytes transferred
df["vL1D"] = ( 64 * (df["rocm:::TCP_TOTAL_CACHE_ACCESSES_sum:device=0"] / df["num samples/calls"]))
# Compute L2 bytes transferred
df["L2"] = ( 64 * ((df["rocm:::TCP_TCC_READ_REQ_sum:device=0"] +
                    df["rocm:::TCP_TCC_WRITE_REQ_sum:device=0"] +
                    df["rocm:::TCP_TCC_ATOMIC_WITH_RET_REQ_sum:device=0"] +
                    df["rocm:::TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum:device=0"]) / df["num samples/calls"]))
# Compute High Bandwidth Memory bytes transferred
df["HBM"] = ((32 * df["rocm:::TCC_EA_RDREQ_32B_sum:device=0"]) +
             (64 * (df["rocm:::TCC_EA_RDREQ_sum:device=0"] -
                  df["rocm:::TCC_EA_RDREQ_32B_sum:device=0"])) +
             (32 * (df["rocm:::TCC_EA_WRREQ_sum:device=0"] -
                  df["rocm:::TCC_EA_WRREQ_64B_sum:device=0"])) +
             (64 *  df["rocm:::TCC_EA_WRREQ_64B_sum:device=0"]))/df["num samples/calls"]

# Compute High Bandwidth Memory bytes transferred bandwidth
df["LDS_bw"] = df["LDS"]/(df["mean"] * 1.0e-9)
df["vL1D_bw"] = df["vL1D"]/(df["mean"] * 1.0e-9)
df["L2_bw"] = df["L2"]/(df["mean"] * 1.0e-9)
df["HBM_bw"] = df["HBM"]/(df["mean"] * 1.0e-9)
# Compute algorithmic intensity (FLOP/byte)
df["AI_LDS"] = df["FLOPS"]/df["LDS_bw"]
df["AI_vL1D"] = df["FLOPS"]/df["vL1D_bw"]
df["AI_L2"] = df["FLOPS"]/df["L2_bw"]
df["AI_HBM"] = df["FLOPS"]/df["HBM_bw"]
# Compute TFLOP/second
df["TFLOP/s"] = (df["FLOPS"]/df["mean"]) * 1.0e-3

# Compute L2 cache utilization
df["L2 Cache Util"] = (df["rocm:::TCC_BUSY_sum:device=0"] /
                       (32 * df["rocm:::GRBM_GUI_ACTIVE:device=0"])) * 100.0
# Compute L2 cache hit ratio
df["L2 Cache Hit Rate %"] = (df["rocm:::TCC_HIT_sum:device=0"] /
                             (df["rocm:::TCC_HIT_sum:device=0"] +
                              df["rocm:::TCC_MISS_sum:device=0"])) * 100.0
# Compute L1 cache utilization
df["vL1D Cache Util"] = (df["rocm:::TCP_GATE_EN2_sum:device=0"] /
                         df["rocm:::TCP_GATE_EN1_sum:device=0"]) * 100.0
# Compute L2 cache hit ratio
df["vL1D Cache Hit Rate %"] = (1.0 - ((df["rocm:::TCP_TCC_READ_REQ_sum:device=0"] +
                                       df["rocm:::TCP_TCC_WRITE_REQ_sum:device=0"] +
                                       df["rocm:::TCP_TCC_ATOMIC_WITH_RET_REQ_sum:device=0"] +
                                       df["rocm:::TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum:device=0"]) /
                                df["rocm:::TCP_TOTAL_CACHE_ACCESSES_sum:device=0"])) * 100.0

# Get instruction mix
df["Total Instructions"] = df["rocm:::SQ_INSTS:device=0"]
df["Vector Instructions %"] = (100.0 * (df["rocm:::SQ_INSTS_VALU:device=0"] / df["rocm:::SQ_INSTS:device=0"]))
df["Vector Pipeline MFMA %"] = (100.0 * (df["rocm:::SQ_INSTS_MFMA:device=0"] / df["rocm:::SQ_INSTS:device=0"]))
df["Scalar Pipeline %"] = (100.0 * ((df["rocm:::SQ_INSTS_SALU:device=0"] + df["rocm:::SQ_INSTS_SMEM:device=0"]) / df["rocm:::SQ_INSTS:device=0"]))
df["Branch Pipeline %"] = (100.0 * (df["rocm:::SQ_INSTS_BRANCH:device=0"] / df["rocm:::SQ_INSTS:device=0"]))
df["Vector Memory Pipeline %"] = (100.0 * (df["rocm:::SQ_INSTS_VMEM:device=0"] / df["rocm:::SQ_INSTS:device=0"]))
df["LDS Pipeline %"] = (100.0 * (df["rocm:::SQ_INSTS_LDS:device=0"] / df["rocm:::SQ_INSTS:device=0"]))
df["Total Memory %"] = (100.0 * ((df["rocm:::SQ_INSTS_VMEM:device=0"] + df["rocm:::SQ_INSTS_LDS:device=0"] + df["rocm:::SQ_INSTS_FLAT_LDS_ONLY:device=0"]) / df["rocm:::SQ_INSTS:device=0"]))

# Wavefronts
df["Average Wavefront Life"] = (4 * (df["rocm:::SQ_WAVE_CYCLES:device=0"] / df["rocm:::SQ_WAVES:device=0"]))
df["Average Dependency Wait Cycles %"] = (100.0 * (4 * (df["rocm:::SQ_WAIT_ANY:device=0"] / df["rocm:::SQ_WAVES:device=0"])) / df["Average Wavefront Life"])
df["Average Issue Wait Cycles %"] = (100.0 * (4 * (df["rocm:::SQ_WAIT_INST_ANY:device=0"] / df["rocm:::SQ_WAVES:device=0"])) / df["Average Wavefront Life"])
df["Average Active Issue Cycles %"] = (100.0 * (4 * (df["rocm:::SQ_ACTIVE_INST_ANY:device=0"] / df["rocm:::SQ_WAVES:device=0"])) / df["Average Wavefront Life"])
df["Average Active Wavefronts"] = (df["rocm:::SQ_THREAD_CYCLES_VALU:device=0"] / df["rocm:::SQ_ACTIVE_INST_VALU:device=0"])

pd.options.display.float_format = "{:,.2f}".format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 60)
print(df[["name","TFLOP/s","AI_HBM"]])
print(df.loc[29])
#print(df.loc[2])


