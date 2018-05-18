#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class Register {
 public:
  using KType = std::chrono::high_resolution_clock::duration(
      const std::vector<const float*>&,
      std::vector<float*>&);

  static Register& get() {
    static Register r;
    return r;
  }

  void registerBenchmark(std::function<KType> impl, uint64_t id) {
    if (kernels.count(id) > 0)
      throw std::invalid_argument{"Multiple kernel ids registered."};
    kernels[id] = impl;
  }

  auto begin() {
    return kernels.begin();
  }

  auto begin() const {
    return kernels.begin();
  }

  auto end() {
    return kernels.end();
  }

  auto end() const {
    return kernels.end();
  }

 private:
  Register() = default;
  std::unordered_map<uint64_t, std::function<KType>> kernels;
};
