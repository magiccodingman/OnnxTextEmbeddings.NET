#include "onnx_text_embeddings.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_handle;
static lib_handle open_library(const char* path) { return LoadLibraryA(path); }
static void* get_symbol(lib_handle lib, const char* name) { return (void*)GetProcAddress(lib, name); }
#else
#include <dlfcn.h>
typedef void* lib_handle;
static lib_handle open_library(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void* get_symbol(lib_handle lib, const char* name) { return dlsym(lib, name); }
#endif

typedef uint32_t (*abi_version_fn)(void);
typedef int32_t (*cosine_f32_fn)(const float*, int32_t, const float*, int32_t, float*);
typedef int32_t (*service_create_fn)(const ote_options*, intptr_t*);
typedef int32_t (*service_destroy_fn)(intptr_t);
typedef int32_t (*service_wait_ready_fn)(intptr_t);
typedef int32_t (*service_dimensions_fn)(intptr_t, int32_t*);
typedef int32_t (*embed_query_fn)(intptr_t, const uint8_t*, size_t, int32_t, ote_buffer*);
typedef void (*buffer_free_fn)(ote_buffer*);
typedef int32_t (*last_error_fn)(uint8_t*, size_t, size_t*);

static void fail_status(last_error_fn last_error, int32_t status, const char* operation) {
    uint8_t message[2048];
    size_t required = 0;
    int32_t error_status = last_error(message, sizeof(message) - 1, &required);
    size_t length = required < sizeof(message) - 1 ? required : sizeof(message) - 1;
    message[length] = 0;
    fprintf(stderr, "%s failed with %d (last-error status %d): %s\n", operation, status, error_status, message);
    exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: native-smoke <shared-library> [--jasper]\n");
        return 2;
    }

    lib_handle lib = open_library(argv[1]);
    if (!lib) {
        fprintf(stderr, "unable to load native library: %s\n", argv[1]);
        return 3;
    }

    abi_version_fn abi_version = (abi_version_fn)get_symbol(lib, "ote_abi_version");
    cosine_f32_fn cosine = (cosine_f32_fn)get_symbol(lib, "ote_vector_cosine_f32");
    last_error_fn last_error = (last_error_fn)get_symbol(lib, "ote_get_last_error");
    if (!abi_version || !cosine || !last_error) {
        fprintf(stderr, "required v1 exports are missing\n");
        return 4;
    }
    if (abi_version() != OTE_ABI_VERSION) {
        fprintf(stderr, "unexpected ABI version\n");
        return 5;
    }

    const float left[] = {1.0f, 0.0f};
    const float right[] = {0.8f, 0.6f};
    float similarity = 0.0f;
    int32_t status = cosine(left, 2, right, 2, &similarity);
    if (status != OTE_OK) fail_status(last_error, status, "ote_vector_cosine_f32");
    if (fabsf(similarity - 0.8f) > 0.001f) {
        fprintf(stderr, "unexpected cosine: %f\n", similarity);
        return 6;
    }
    printf("PASS C ABI v1 + float32 cosine smoke.\n");

    if (argc >= 3 && strcmp(argv[2], "--jasper") == 0) {
        service_create_fn create_service = (service_create_fn)get_symbol(lib, "ote_service_create");
        service_destroy_fn destroy_service = (service_destroy_fn)get_symbol(lib, "ote_service_destroy");
        service_wait_ready_fn wait_ready = (service_wait_ready_fn)get_symbol(lib, "ote_service_wait_ready");
        service_dimensions_fn dimensions_fn = (service_dimensions_fn)get_symbol(lib, "ote_service_model_dimensions");
        embed_query_fn embed_query = (embed_query_fn)get_symbol(lib, "ote_embed_query_json");
        buffer_free_fn buffer_free = (buffer_free_fn)get_symbol(lib, "ote_buffer_free");
        if (!create_service || !destroy_service || !wait_ready || !dimensions_fn || !embed_query || !buffer_free) {
            fprintf(stderr, "service exports are missing\n");
            return 7;
        }

        ote_options options;
        memset(&options, 0, sizeof(options));
        options.struct_size = (uint32_t)sizeof(options);
        options.abi_version = OTE_ABI_VERSION;
        options.model_precision = OTE_JASPER_INT8;
        intptr_t handle = 0;
        status = create_service(&options, &handle);
        if (status != OTE_OK) fail_status(last_error, status, "ote_service_create");
        status = wait_ready(handle);
        if (status != OTE_OK) fail_status(last_error, status, "ote_service_wait_ready");
        int32_t dimensions = 0;
        status = dimensions_fn(handle, &dimensions);
        if (status != OTE_OK) fail_status(last_error, status, "ote_service_model_dimensions");
        if (dimensions != 2048) {
            fprintf(stderr, "expected 2048 Jasper dimensions, got %d\n", dimensions);
            return 8;
        }
        const char* query = "restore a PostgreSQL database backup";
        ote_buffer output = {0};
        status = embed_query(handle, (const uint8_t*)query, strlen(query), OTE_VECTOR_FLOAT32, &output);
        if (status != OTE_OK) fail_status(last_error, status, "ote_embed_query_json");
        if (!output.data || output.length < 32) {
            fprintf(stderr, "native query embedding JSON was unexpectedly empty\n");
            return 9;
        }
        buffer_free(&output);
        status = destroy_service(handle);
        if (status != OTE_OK) fail_status(last_error, status, "ote_service_destroy");
        printf("PASS C ABI Native AOT Jasper inference smoke.\n");
    }

    return 0;
}
