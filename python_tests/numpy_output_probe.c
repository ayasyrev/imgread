/* Test-only NumPy memory policy: observe RGB allocation/free and inject failure. */
#define PY_SSIZE_T_CLEAN
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>

static size_t target_size;
static int fail_next, notified;
static unsigned long allocations, releases, zero_allocations;
static PyObject *callback;

static int check_allocation(size_t size) {
    if (size != target_size || notified) return 0;
    notified = 1;
    if (callback && callback != Py_None) {
        PyObject *result = PyObject_CallNoArgs(callback);
        if (!result) return -1;
        Py_DECREF(result);
    }
    return fail_next ? -1 : 0;
}

static void *probe_malloc(void *ctx, size_t size) {
    if (check_allocation(size)) return NULL;
    void *ptr = malloc(size);
    if (ptr && size == target_size) allocations++;
    return ptr;
}

static void *probe_calloc(void *ctx, size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) return NULL;
    size_t total = count * size;
    if (check_allocation(total)) return NULL;
    void *ptr = calloc(count, size);
    if (ptr && total == target_size) { allocations++; zero_allocations++; }
    return ptr;
}

static void *probe_realloc(void *ctx, void *ptr, size_t size) {
    return realloc(ptr, size);
}

static void probe_free(void *ctx, void *ptr, size_t size) {
    if (ptr && size == target_size) releases++;
    free(ptr);
}

static PyDataMem_Handler handler = {
    "imgread_numpy_output_test", 1,
    {NULL, probe_malloc, probe_calloc, probe_realloc, probe_free}
};

static PyObject *install(PyObject *self, PyObject *args) {
    Py_ssize_t size;
    PyObject *cb = Py_None;
    int fail = 0;
    if (!PyArg_ParseTuple(args, "n|Op", &size, &cb, &fail)) return NULL;
    if (size <= 0) return PyErr_Format(PyExc_ValueError, "positive size required");
    target_size = (size_t)size;
    fail_next = fail;
    notified = 0;
    allocations = releases = zero_allocations = 0;
    Py_XSETREF(callback, Py_NewRef(cb));
    PyObject *capsule = PyCapsule_New(&handler, "mem_handler", NULL);
    if (!capsule) return NULL;
    PyObject *old = PyDataMem_SetHandler(capsule);
    Py_DECREF(capsule);
    return old;
}

static PyObject *restore(PyObject *self, PyObject *old) {
    PyObject *previous = PyDataMem_SetHandler(old);
    if (!previous) return NULL;
    Py_DECREF(previous);
    Py_CLEAR(callback);
    Py_RETURN_NONE;
}

static PyObject *stats(PyObject *self, PyObject *unused) {
    return Py_BuildValue("(kkki)", allocations, releases, zero_allocations, notified);
}

static PyMethodDef methods[] = {
    {"install", install, METH_VARARGS, NULL},
    {"restore", restore, METH_O, NULL},
    {"stats", stats, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_numpy_output_probe", NULL, -1, methods
};
PyMODINIT_FUNC PyInit__numpy_output_probe(void) {
    import_array();
    return PyModule_Create(&module);
}
