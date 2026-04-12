#include <rais/scheduler.hpp>
#include <rais/task.hpp>

#include <pybind11/pybind11.h>

#include <memory>
#include <utility>

namespace py = pybind11;

namespace {

using PyFnHolder = std::shared_ptr<py::function>;

PyFnHolder hold_py_function(py::function fn) {
    return PyFnHolder(new py::function(std::move(fn)), [](py::function* fn_ptr) {
        py::gil_scoped_acquire acquire;
        delete fn_ptr;
    });
}

void run_python_callback(const PyFnHolder& fn) {
    py::gil_scoped_acquire acquire;
    try {
        (*fn)();
    } catch (const py::error_already_set& e) {
        PyErr_WriteUnraisable(e.value().ptr());
    }
}

} // namespace

PYBIND11_MODULE(rais, m) {
    m.doc() = "Python bindings for the RAIS scheduler";

    py::enum_<rais::Lane>(m, "Lane")
        .value("INTERACTIVE", rais::Lane::Interactive)
        .value("BACKGROUND", rais::Lane::Background)
        .value("BULK", rais::Lane::Bulk)
        .value("GPU", rais::Lane::GPU)
        .value("IO", rais::Lane::IO)
        .export_values();

    py::enum_<rais::ShutdownPolicy>(m, "ShutdownPolicy")
        .value("DRAIN", rais::ShutdownPolicy::Drain)
        .value("CANCEL", rais::ShutdownPolicy::Cancel)
        .export_values();

    py::class_<rais::TaskHandle>(m, "TaskHandle")
        .def("wait", &rais::TaskHandle::wait, py::call_guard<py::gil_scoped_release>())
        .def("cancel", &rais::TaskHandle::cancel)
        .def("done", &rais::TaskHandle::done);

    py::class_<rais::Scheduler>(m, "Scheduler")
        .def(py::init([](size_t num_workers, size_t io_thread_count) {
                 rais::SchedulerConfig cfg{};
                 cfg.num_workers = num_workers;
                 cfg.io_thread_count = io_thread_count;
                 return std::make_unique<rais::Scheduler>(cfg);
             }),
             py::arg("num_workers") = 0, py::arg("io_thread_count") = 2)
        .def("submit",
             [](rais::Scheduler& self, py::function fn, rais::Lane lane) {
                 auto holder = hold_py_function(std::move(fn));
                 return self.submit([holder]() { run_python_callback(holder); }, lane);
             },
             py::arg("fn"), py::arg("lane") = rais::Lane::Background)
        .def("shutdown", &rais::Scheduler::shutdown,
             py::arg("policy") = rais::ShutdownPolicy::Drain)
        .def("lane_count", &rais::Scheduler::lane_count, py::arg("lane"))
        .def("deadline_misses", &rais::Scheduler::deadline_misses);
}
