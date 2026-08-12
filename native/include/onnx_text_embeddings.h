#ifndef ONNX_TEXT_EMBEDDINGS_H
#define ONNX_TEXT_EMBEDDINGS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OTE_ABI_VERSION 1u

typedef enum ote_status {
    OTE_OK = 0,
    OTE_INVALID_ARGUMENT = 1,
    OTE_BUFFER_TOO_SMALL = 2,
    OTE_INVALID_HANDLE = 3,
    OTE_MODEL_ERROR = 4,
    OTE_QUERY_TOO_LONG = 5,
    OTE_EMBEDDING_SPACE_MISMATCH = 6,
    OTE_SERIALIZATION_ERROR = 7,
    OTE_OUT_OF_MEMORY = 8,
    OTE_INTERNAL_ERROR = 255
} ote_status;

typedef enum ote_vector_format {
    OTE_VECTOR_INT4 = 1,
    OTE_VECTOR_INT8 = 2,
    OTE_VECTOR_FLOAT16 = 3,
    OTE_VECTOR_FLOAT32 = 4
} ote_vector_format;

typedef enum ote_jasper_precision {
    OTE_JASPER_INT8 = 1,
    OTE_JASPER_INT4 = 2,
    OTE_JASPER_FLOAT32 = 3
} ote_jasper_precision;

typedef struct ote_options {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t model_precision;
    int32_t document_max_tokens;
    int32_t query_max_tokens;
    int32_t model_instance_count;
    int32_t threads_per_model;
    int32_t concurrent_requests_per_model;
    int32_t queue_capacity;
    int32_t document_vector_format;
    int32_t query_vector_format;
} ote_options;

typedef struct ote_buffer {
    uint8_t* data;
    size_t length;
} ote_buffer;

typedef struct ote_query_token_count {
    int32_t source_token_count;
    int32_t input_token_count;
    int32_t query_max_tokens;
    int32_t model_max_tokens;
    int32_t has_model_max_tokens;
    int32_t fits;
} ote_query_token_count;

uint32_t ote_abi_version(void);
int32_t ote_get_last_error(uint8_t* buffer, size_t buffer_length, size_t* required_length);
int32_t ote_service_create(const ote_options* options, intptr_t* output_handle);
int32_t ote_service_destroy(intptr_t handle);
int32_t ote_service_wait_ready(intptr_t handle);
int32_t ote_service_model_dimensions(intptr_t handle, int32_t* dimensions);
int32_t ote_count_tokens(intptr_t handle, const uint8_t* text, size_t text_length, int32_t* token_count);
int32_t ote_count_query_tokens(intptr_t handle, const uint8_t* text, size_t text_length, ote_query_token_count* output);
int32_t ote_embed_query_json(intptr_t handle, const uint8_t* text, size_t text_length, int32_t vector_format, ote_buffer* output);
int32_t ote_embed_document_json(intptr_t handle, const uint8_t* text, size_t text_length, int32_t vector_format, ote_buffer* output);
int32_t ote_vector_convert(const uint8_t* vector_bytes, size_t vector_length, int32_t target_format, ote_buffer* output);
int32_t ote_vector_cosine(const uint8_t* left, size_t left_length, const uint8_t* right, size_t right_length, float* similarity);
int32_t ote_vector_cosine_f32(const float* left, int32_t left_dimensions, const float* right, int32_t right_dimensions, float* similarity);
int32_t ote_reduce_query_json(const uint8_t* json, size_t json_length, int32_t output_dimensions, int32_t output_format, ote_buffer* output);
int32_t ote_reduce_text_embedding_json(const uint8_t* json, size_t json_length, int32_t output_dimensions, int32_t output_format, ote_buffer* output);
int32_t ote_combine_to_single_json(const uint8_t* json, size_t json_length, int32_t output_dimensions, int32_t output_format, ote_buffer* output);
void ote_buffer_free(ote_buffer* buffer);

#ifdef __cplusplus
}
#endif

#endif
