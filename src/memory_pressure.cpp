#include <rais/memory_pressure.hpp>

namespace rais {

MemoryMonitor::MemoryMonitor(MetalBufferPool& pool,
                             float warning_fraction,
                             float critical_fraction)
    : pool_(pool)
    , warning_fraction_(warning_fraction)
    , critical_fraction_(critical_fraction) {}

MemoryPressure MemoryMonitor::check() const {
    size_t budget = pool_.memory_budget();
    if (budget == 0) return MemoryPressure::Normal;

    size_t allocated = pool_.total_allocated_bytes();
    float ratio = static_cast<float>(allocated) / static_cast<float>(budget);

    if (ratio >= critical_fraction_) return MemoryPressure::Critical;
    if (ratio >= warning_fraction_)  return MemoryPressure::Warning;
    return MemoryPressure::Normal;
}

bool MemoryMonitor::under_pressure() const {
    return check() != MemoryPressure::Normal;
}

} // namespace rais
