#pragma once

#include <rais/metal_allocator.hpp>

namespace rais {

enum class MemoryPressure { Normal, Warning, Critical };

/// Monitors GPU buffer pool utilization relative to its memory budget.
///
/// Thresholds are expressed as fractions of the pool's memory_budget().
/// If no budget is set (budget == 0), always reports Normal.
class MemoryMonitor {
public:
    /// warning_fraction: transition to Warning when allocated >= budget * fraction
    /// critical_fraction: transition to Critical when allocated >= budget * fraction
    MemoryMonitor(MetalBufferPool& pool,
                  float warning_fraction = 0.75f,
                  float critical_fraction = 0.90f);

    MemoryPressure check() const;

    /// Convenience: true if check() returns Warning or Critical.
    bool under_pressure() const;

private:
    MetalBufferPool& pool_;
    float warning_fraction_;
    float critical_fraction_;
};

} // namespace rais
