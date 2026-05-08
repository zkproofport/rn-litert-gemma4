// rust_alloc_shim.c
//
// Provides Rust runtime allocator shim symbols (__rust_alloc, __rust_dealloc,
// __rust_realloc, __rust_alloc_zeroed, __rust_alloc_error_handler) that are
// normally synthesized by `cargo` or `rust_binary` link step. When linking
// Rust .rlib files into a non-Rust binary (cc_library archive consumed by
// Cocoapods/Xcode), these symbols are missing and link fails with:
//   Undefined symbols: ___rust_alloc, ___rust_dealloc, etc.
//
// We provide a System allocator implementation (posix_memalign-based).

#include <stdlib.h>
#include <string.h>

void* __rust_alloc(size_t size, size_t align) {
    void* ptr = NULL;
    if (align < sizeof(void*)) align = sizeof(void*);
    if (posix_memalign(&ptr, align, size) != 0) return NULL;
    return ptr;
}

void __rust_dealloc(void* ptr, size_t size, size_t align) {
    (void)size;
    (void)align;
    free(ptr);
}

void* __rust_realloc(void* ptr, size_t old_size, size_t align, size_t new_size) {
    (void)old_size;
    if (align <= sizeof(void*)) {
        return realloc(ptr, new_size);
    }
    void* new_ptr = __rust_alloc(new_size, align);
    if (!new_ptr) return NULL;
    if (ptr) {
        size_t copy = old_size < new_size ? old_size : new_size;
        memcpy(new_ptr, ptr, copy);
        free(ptr);
    }
    return new_ptr;
}

void* __rust_alloc_zeroed(size_t size, size_t align) {
    void* ptr = __rust_alloc(size, align);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

void __rust_alloc_error_handler(size_t size, size_t align) {
    (void)size;
    (void)align;
    abort();
}

const unsigned char __rust_alloc_error_handler_should_panic = 0;
const unsigned char __rust_no_alloc_shim_is_unstable = 0;
