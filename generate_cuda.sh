#!/bin/bash




retriever=${1}
protofile=${2}
benchmaker_header_dir=${3}
cuda_header_dir=${4}
export benchmaker_header_dir
export cuda_header_dir

com="${retriever} --input=${protofile} "

export com

s_1=$(($(${com} --size) - 1)) 

 
function pp {
    local id=${1}
    c="${com} --idx=${id}"


    local source=$(${c})
    local specialized_name=$(${c} --sname)
    local params=$(${c} --params)
    local noutputs=$(${c} --noutputs)
    local ninputs=$(${c} --ninputs)
    local id=$(${c} --id)
    local grid=$(${c} --grid)
    local block=$(${c} --block)

    inputs="inputs[0]"

    for i in `seq 1 $((${ninputs}-1))`; do
        inputs="${inputs}, inputs[${i}]"
    done

    outputs="outputs[0]"
    for o in `seq 1 $((${noutputs}-1))`; do
        outputs="${outputs}, outputs[${o}]"
    done

    printf "#include \"benchmark_register.hpp\"
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
namespace{


#define Check(condition)                                        \
  do {                                                          \
    cudaError_t result = condition;                             \
    if (result != cudaSuccess) {                                \
      std::stringstream ss;                                     \
      ss << \"Error at: \" << __FILE__ << \":\" << __LINE__ << \": \" \
         << cudaGetErrorString(result);                         \
      throw std::runtime_error(ss.str().c_str());               \
    }                                                           \
  } while (0)

${source}

auto dummy = [](){
  Register::get().registerBenchmark([]
  (const std::vector<const float*>& inputs, std::vector<float*>& outputs){
    dim3 grid{${grid}};
    dim3 block{${block}};
    cudaEvent_t start, stop;
    Check(cudaEventCreate(&start));
    Check(cudaEventCreate(&stop));
    Check(cudaEventRecord(start, 0));
      ${specialized_name}<<<grid, block>>>(${params}, ${outputs}, ${inputs}); 
      Check(cudaEventRecord(stop, 0));
    Check(cudaEventSynchronize(stop));
    float ms = 0.0f;
    Check(cudaEventElapsedTime(&ms, start, stop));
    Check(cudaEventDestroy(start));
    Check(cudaEventDestroy(stop));
    return std::chrono::microseconds{static_cast<uint64_t>(ms*1000)}; }, ${id});
    return 0;
}();

}\n"
}

export -f pp

function compile {
    cuda=$(pp ${1})
    fname=$(mktemp /tmp/cuda_sourceXXXXXX.cu)
    printf "${cuda}\n" > ${fname}
    #nvcc ${fname} -I${benchmaker_header_dir} -I${cuda_header_dir} -c -O3 --use_fast_math -std=c++14 -o cuda_${1}.o 
}

export -f compile

seq 0 ${s_1} | parallel -a - compile {}
