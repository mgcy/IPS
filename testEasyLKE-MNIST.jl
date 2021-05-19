# include("G:/Code/EasyLKE.jl/EasyLKE.jl")
# include("G:/Code/EasySRC.jl/EasyData.jl")
# include("G:/Code/EasySRC.jl/EasySRC.jl")
include("D:/Academic/LKDL/LKD/EasyLKE.jl/EasyLKE.jl")
include("D:/Academic/LKDL/LKD/EasySRC.jl/EasyData.jl")
include("D:/Academic/LKDL/LKD/EasySRC.jl/EasySRC.jl")

using CSV, DelimitedFiles
using Main.EasyLKE
using Main.EasyData
using Main.EasySRC
using Random
using KernelFunctions
using LinearAlgebra
using PyCall
using NPZ
np = pyimport("numpy")

# Parameters for RUnning Our desired Experiment
use_saved1   = false#false#false#true#true#%false
use_saved2   = false#true#false#true#true

baseline_testing = true
t_alone_testing  = false
apd              = 1


base_path   = "D:/Academic/LKDL/LKD/Models"
model_name  = "LKE-MNIST-12-29-20.jld"#"LKE-UXOs-12-14-20.csv"#
mat_name    = "LKE-MNIST-12-29-20.csv"

# Names of some model and embedding parameters that will be saved
V1S1Z1name  = string(base_path,"V1S1Z1-",mat_name)
V2S2Z2name  = string(base_path,"V2S2Z2-",mat_name)
V3S3Z3name  = string(base_path,"V3S3Z3-",mat_name)

Model1name  = string(base_path,"M1-",model_name)
Model2name  = string(base_path,"M2-IK-TREX-",model_name)
Model3name  = string(base_path,"M3-IPS-",model_name)
# Importing Datasets (taking only a fraction of the total available)

train_X = np.load("train_X.npy", allow_pickle=true)
train_X =  convert(Matrix{Float64}, train_X)
train_X = transpose(train_X)
train_Y = np.load("train_Y.npy", allow_pickle=true)

test_X = np.load("test_aug_KNN_X.npy", allow_pickle=true)
test_Y = np.load("test_Y.npy", allow_pickle=true)

#Normalize Columns of datasets between 0-1
train_X = train_X./-100
test_X = test_X./-100

# Training percentages for month 1
# tp = 1
# num_tp = floor(Int, size(train_X)[2]*tp)
# num_tp = 1185


# Selecting Important Samples from dataset 1 (MNIST)
# p = 1194-300
# Ztr1ind  = randperm(size(train_X,2))[1:p]
# Ztr1 = train_X[:,Ztr1ind]
# ytr1 = train_Y[Ztr1ind]
#
# Ztr2ind = setdiff(Array((1:size(train_X,2))), Ztr1ind)
# Ztr2 = train_X[:,Ztr2ind]
# ytr2 = train_Y[Ztr2ind]

# my scheme

Ztr1 = train_X
ytr1 = train_Y

Zte1 = test_X
yte1 = test_Y


# convert int32 to int64
ytr1 = convert(Vector{Int64},ytr1)
yte1 = convert(Array{Int64},yte1)

display("Data Separated into Training and Testing Sets...")

# Embedding Samples using a Random set of Samples for ZR
# c is the number of important samples (the column dimension of ZR)
c = 800
# Selecting Important Samples from dataset 1 (MNIST)
ZR1ind  = randperm(size(Ztr1,2))[1:c]
ZR1     = Ztr1[:,ZR1ind]

# Specifying Kernel Function and Generating Embedding Map Components
kfnc = SqExponentialKernel()#
kfnc = transform(kfnc,4.0)
# Some Other Kernel Choices:  LaplacianKernel() MaternKernel(), Matern32Kernel(), Matern52Kernel() LinearKernel(c=0.5)#PolynomialKernel(c=2.0,d=2.0)

# If we aren't using a saved embedding then re-solve the embedding
if use_saved1
    V1S1Z1  = readdlm(V1S1Z1name,',')
    c       = size(V1S1Z1,1)
    V1      = V1S1Z1[:,1:c]
    S1      = vec(V1S1Z1[:,c+1])
    ZR1     = V1S1Z1[:,c+2:end]'
else
    V1,S1,Ctr1  = EasyLKE.embedLKE(Ztr1, ZR1, kfnc)
    V1S1Z1      = cat(V1,S1,ZR1',dims=2)
    writedlm(V1S1Z1name,  V1S1Z1, ',')
end

# Generating the C matrices for other training and testing sets
Ctr1=kernelmatrix(kfnc,Ztr1,ZR1,obsdim=2)

# Selecting the rank of the subspace that samples will be projected into
k = 500 # 300#400

Vk1=V1[:,1:k]
Sk1=S1[1:k]
# Generating the Vitrual Samples
Ftr1=diagm(Sk1)^(-1/2)*Vk1'*Ctr1'
Utr1=Ftr1

# Parameters for Baseline Model
struct params
    K::Int64
    max_iter::Int64
    tau::Int64
    SA::Float64
    data_init::Bool
    des_avg_err::Float64
end

tau             = 10 #15
max_iter        = 30#20#30
K               = 200
des_avg_err     = 1e-6
learning_params = params(K,max_iter,tau,tau/K,true,des_avg_err)
learning_method = "KSVD"

# Generate Baseline Model using MNIST training only
if !use_saved1
    Model3=EasySRC.genSRCModel(learning_method,learning_params,Utr1,ytr1)
    # Saving Generated Model
    EasySRC.saveSRCModel(Model3name,Model3)
else
    Model3=EasySRC.genSRCModel(Model3name)
end

# Make Inferences on the MNIST test data
if baseline_testing
    # Yifan: try first month
    pred_y = zeros(15,260)
    actu_y = zeros(15,260)
    for month=1:15
    # for month=1:1
        Zte_tmp = Zte1[month,:,:]
        Zte_tmp = transpose(Zte_tmp)
        yte_tmp = yte1[month,:]

        Cte_tmp = kernelmatrix(kfnc,Zte_tmp,ZR1,obsdim=2)
        Fte_tmp = diagm(Sk1)^(-1/2)*Vk1'*Cte_tmp'
        Ute_tmp = Fte_tmp

        # println("MNIST Performance, Model 1")
        println("Baseline Testing: ")
        println("Testing on month: ", month)
        stats,decs=EasySRC.classify(Ute_tmp,Model3,apd)
        # stats,decs=EasySRC.classify(Utr1,Model3,apd)
        deca = decs .- 1
        acc = sum(deca.==yte_tmp)/length(yte_tmp)
        # acc = sum(deca.==ytr1)/length(ytr1)
        pred_y[month,:] = deca
        actu_y[month,:] = yte_tmp
        display(acc)
    end
    npzwrite("actu_aug_try_k9_y.npy", actu_y)
    npzwrite("pred_aug_try_k9_y.npy", pred_y)
    # npzwrite("actu_baseline_y.npy", actu_y)
    # npzwrite("pred_baseline_y.npy", pred_y)
else
    display("Baseline Testing Skipped!")
end

#=
## Insiti learning
pred_y = zeros(15,260)
actu_y = zeros(15,260)

#Ztr2 = train_X[:,num_tp:end]
#ytr2 = train_Y[num_tp:end]
#ytr2 = convert(Vector{Int64},ytr2)

for month =2:15
    # fetch testing data
    Zte1 = test_X[month-1,:,:]
    Zte1 = transpose(Zte1)
    yte1 = test_Y[month-1,:]
    yte1 = convert(Vector{Int64},yte1)

    Zte2 = test_X[month,:,:]
    Zte2 = transpose(Zte2)
    yte2 = test_Y[month,:]
    yte2 = convert(Vector{Int64},yte2)

    # fetch tr2
    if month in [4,7,11,14]

        n0 = 30
        Ztr0ind  = randperm(size(yte2,1))[1:n0]
        global Ztr2 = Zte2[:,Ztr0ind]
        global ytr2 = yte2[Ztr0ind]

    end

    # Ztr2 = Zte2[:,month*15:month*15+15]
    # ytr2 = yte2[month*15:month*15+15]


    ## Update Embedding using limited amount of USPS
    b       = 200#size(Zte2,2)#345
    ZR2ind  = randperm(size(Zte1,2))[1:b]
    ZRnew   = Zte1[:,ZR2ind]

    if use_saved2
        V2S2Z2  = readdlm(V2S2Z2name,',')
        c       = size(V2S2Z2,1)
        V2      = V2S2Z2[:,1:c]
        S2      = vec(V2S2Z2[:,c+1])
        ZR2     = V2S2Z2[:,c+2:end]'
        k₂      = k#+20##k+60c
    else
        V2,S2,ZR2   = EasyLKE.incrementLKE(V1, S1, ZR1, ZRnew, kfnc)
        # display("Embedding Update Complete!")
        # Saving Updated embedding components
        V2S2Z2      = cat(V2,S2,ZR2',dims=2)

        #k = size(V2S2Z2,1)

        k₂  = k#+20##k+60
        writedlm(V2S2Z2name,  V2S2Z2, ',')
    end

    # Generate New Kernel Approximation Wtilde (Wt) and augmented kernel matrix W₊
    Wt  = V2*diagm(S2)*V2'
    W₊  = kernelmatrix(kfnc,ZR1,ZR2,obsdim=2)

    # Select New Subspace rank and the corresponding eigenvalues/vectors of Wt
    Vk2 = V2[:,1:k₂]
    Sk2 = S2[1:k₂]

    # Generate Dictionary Transforming Matrix to update dictionaries to new embedding
    T = diagm(Sk2)^(-1/2)*Vk2'*Wt*W₊'*(W₊*W₊')^(-1)*Vk1*diagm(Sk1)^(1/2)
    # Generating the Updated C matrices using the new ZR (important samples) matrix.
    Cte2    = kernelmatrix(kfnc,Zte2,ZR2,obsdim=2)
    # Generating the Corresponding F matrices for updated embedding
    Ute2    = diagm(Sk2)^(-1/2)*Vk2'*Cte2'



    # Transform Old Model Dictionaries to new space
    Dold    = Model3.D
    n_dicts = length(Dold)
    Dnew    =  Array{Float64,2}[]#zeros(k₂,K,n_dicts)

    for class in 1:n_dicts
        DD              = T*Dold[class]
        push!(Dnew,DD)
    end
    # Update Model dictionaries to new embedding
    Model_T=EasySRC.SRCModel(Dnew,Model3.n_classes,Model3.sparsity, Model3.des_avg_err)
    # global Model3=EasySRC.SRCModel(Dnew,Model3.n_classes,Model3.sparsity, Model3.des_avg_err)

    display("Dictionaries Have Been Transformed to New Embedding")
    ## Update Model using limited amount of testing data
    # for IKSVD
    # d =40
    Ctr2    = kernelmatrix(kfnc,Ztr2,ZR2,obsdim=2)
    Utr2    = diagm(Sk2)^(-1/2)*Vk2'*Ctr2'

    # Utr3 = Ute2[:,1:d]
    # ytr3 = yte2[1:d]
    K1    =   100#30#10     # Number of additional atoms to add.


    # Train Incremental update to dictionaries using IDUO or IKSVD
    Model4  = EasySRC.incIKSVD2(Model_T,Utr2,ytr2,ytr1,learning_params,K1)

    println("Insitu Testing: ")
    println("Testing on month: ", month)
    stats,decs=EasySRC.classify(Ute2,Model4,apd)
    deca = decs .- 1
    acc = sum(deca.==yte2)/length(yte2)
    pred_y[month,:] = deca
    actu_y[month,:] = yte2
    display(acc)
end
# npzwrite("actu_test_y.npy", actu_y)
# npzwrite("pred_test_y.npy", pred_y)
npzwrite("actu_insitu_aug_KNN_y.npy", actu_y)
npzwrite("pred_insitu_aug_KNN_y.npy", pred_y)
=#
