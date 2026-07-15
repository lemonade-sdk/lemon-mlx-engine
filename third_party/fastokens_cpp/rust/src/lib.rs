//! C FFI for [fastokens](https://github.com/crusoecloud/fastokens).
//!
//! Mirrors the surface used by lemon-mlx-engine's `Tokenizer` wrapper:
//! load from `tokenizer.json`, encode, decode, vocab lookups.

use std::os::raw::c_char;
use std::slice;

use fastokens::Tokenizer;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    decode_str: String,
    id_to_token_result: String,
}

#[repr(C)]
pub struct FastokensEncodeResult {
    token_ids: *mut i32,
    len: usize,
}

impl TokenizerWrapper {
    fn from_json_str(json: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("invalid tokenizer.json: {e}"))?;
        let tokenizer =
            Tokenizer::from_json(value).map_err(|e| format!("failed to build tokenizer: {e}"))?;
        Ok(Self {
            tokenizer,
            decode_str: String::new(),
            id_to_token_result: String::new(),
        })
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode_with_special_tokens(text, add_special_tokens)
            .map_err(|e| e.to_string())
    }

    fn decode(&mut self, ids: &[u32], skip_special_tokens: bool) -> Result<(), String> {
        self.decode_str = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Convert a UTF-8 C buffer into a Rust `&str`. Empty/`null` → `""`.
unsafe fn cstr_to_str<'a>(ptr: *const c_char, len: usize) -> &'a str {
    if ptr.is_null() || len == 0 {
        return "";
    }
    std::str::from_utf8(slice::from_raw_parts(ptr as *const u8, len)).unwrap_or("")
}

#[no_mangle]
pub extern "C" fn fastokens_new_from_str(json: *const c_char, len: usize) -> *mut TokenizerWrapper {
    unsafe {
        let json_str = cstr_to_str(json, len);
        match TokenizerWrapper::from_json_str(json_str) {
            Ok(wrapper) => Box::into_raw(Box::new(wrapper)),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

#[no_mangle]
pub extern "C" fn fastokens_encode(
    handle: *mut TokenizerWrapper,
    data: *const c_char,
    len: usize,
    add_special_tokens: i32,
    out_result: *mut FastokensEncodeResult,
) {
    unsafe {
        if handle.is_null() || out_result.is_null() {
            return;
        }
        let text = cstr_to_str(data, len);
        let ids_u32 = match (*handle).encode(text, add_special_tokens != 0) {
            Ok(ids) => ids,
            Err(_) => {
                *out_result = FastokensEncodeResult {
                    token_ids: std::ptr::null_mut(),
                    len: 0,
                };
                return;
            }
        };
        let mut ids_i32: Vec<i32> = ids_u32.into_iter().map(|id| id as i32).collect();
        let out_len = ids_i32.len();
        let ptr = ids_i32.as_mut_ptr();
        std::mem::forget(ids_i32);
        *out_result = FastokensEncodeResult {
            token_ids: ptr,
            len: out_len,
        };
    }
}

#[no_mangle]
pub extern "C" fn fastokens_free_encode_result(result: *mut FastokensEncodeResult) {
    unsafe {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        if !r.token_ids.is_null() && r.len > 0 {
            drop(Vec::from_raw_parts(r.token_ids, r.len, r.len));
        }
        r.token_ids = std::ptr::null_mut();
        r.len = 0;
    }
}

#[no_mangle]
pub extern "C" fn fastokens_decode(
    handle: *mut TokenizerWrapper,
    data: *const u32,
    len: usize,
    skip_special_tokens: i32,
) {
    unsafe {
        if handle.is_null() {
            return;
        }
        let ids = if data.is_null() || len == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(data, len)
        };
        let _ = (*handle).decode(ids, skip_special_tokens != 0);
    }
}

#[no_mangle]
pub extern "C" fn fastokens_get_decode_str(
    handle: *mut TokenizerWrapper,
    data: *mut *const c_char,
    len: *mut usize,
) {
    unsafe {
        if handle.is_null() || data.is_null() || len.is_null() {
            return;
        }
        let s = &(*handle).decode_str;
        *data = s.as_ptr() as *const c_char;
        *len = s.len();
    }
}

#[no_mangle]
pub extern "C" fn fastokens_get_vocab_size(handle: *mut TokenizerWrapper, size: *mut usize) {
    unsafe {
        if handle.is_null() || size.is_null() {
            return;
        }
        *size = (*handle).tokenizer.vocab_size();
    }
}

#[no_mangle]
pub extern "C" fn fastokens_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    data: *mut *const c_char,
    len: *mut usize,
) {
    unsafe {
        if handle.is_null() || data.is_null() || len.is_null() {
            return;
        }
        let wrapper = &mut *handle;
        match wrapper.tokenizer.id_to_token(id) {
            Some(token) => {
                wrapper.id_to_token_result = token.to_string();
                *data = wrapper.id_to_token_result.as_ptr() as *const c_char;
                *len = wrapper.id_to_token_result.len();
            }
            None => {
                wrapper.id_to_token_result.clear();
                *data = wrapper.id_to_token_result.as_ptr() as *const c_char;
                *len = 0;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn fastokens_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const c_char,
    len: usize,
    id: *mut i32,
) {
    unsafe {
        if handle.is_null() || id.is_null() {
            return;
        }
        let token_str = cstr_to_str(token, len);
        *id = match (*handle).tokenizer.token_to_id(token_str) {
            Some(v) => v as i32,
            None => -1,
        };
    }
}

#[no_mangle]
pub extern "C" fn fastokens_free(handle: *mut TokenizerWrapper) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}
