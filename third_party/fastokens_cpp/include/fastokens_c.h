/*!
 * \file fastokens_c.h
 * \brief C binding to the crusoecloud/fastokens Rust tokenizer.
 */
#ifndef FASTOKENS_C_H_
#define FASTOKENS_C_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* FastokensHandle;

typedef struct {
    int* token_ids;
    size_t len;
} FastokensEncodeResult;

/* Create from an in-memory tokenizer.json blob. Returns NULL on failure. */
FastokensHandle fastokens_new_from_str(const char* json, size_t len);

/* Encode text. add_special_tokens: non-zero to apply post-processor specials. */
void fastokens_encode(FastokensHandle handle, const char* data, size_t len,
                      int add_special_tokens, FastokensEncodeResult* result);

void fastokens_free_encode_result(FastokensEncodeResult* result);

/* Decode token IDs. Result is stored in the handle; use fastokens_get_decode_str. */
void fastokens_decode(FastokensHandle handle, const uint32_t* data, size_t len,
                      int skip_special_tokens);

void fastokens_get_decode_str(FastokensHandle handle, const char** data, size_t* len);

void fastokens_get_vocab_size(FastokensHandle handle, size_t* size);

void fastokens_id_to_token(FastokensHandle handle, uint32_t id, const char** data,
                           size_t* len);

/* Stores -1 to *id if the token is not in the vocab. */
void fastokens_token_to_id(FastokensHandle handle, const char* token, size_t len,
                           int32_t* id);

void fastokens_free(FastokensHandle handle);

#ifdef __cplusplus
}
#endif
#endif /* FASTOKENS_C_H_ */
