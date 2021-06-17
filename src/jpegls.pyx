cimport numpy as np
from libc.stdint cimport *
import numpy as np
import math

cdef extern from 'charls/charls.h':
	ctypedef enum charls_jpegls_errc:
		CHARLS_JPEGLS_ERRC_SUCCESS = 0

	ctypedef enum charls_interleave_mode:
		CHARLS_INTERLEAVE_MODE_NONE = 0,
		CHARLS_INTERLEAVE_MODE_LINE = 1,
		CHARLS_INTERLEAVE_MODE_SAMPLE = 2

	ctypedef struct charls_frame_info:
		uint32_t width;
		uint32_t height;
		int32_t bits_per_sample;
		int32_t component_count;

	ctypedef struct charls_jpegls_encoder:
		pass

	ctypedef struct charls_jpegls_decoder:
		pass

	cdef const char* charls_get_error_message(charls_jpegls_errc)

	cdef charls_jpegls_encoder* charls_jpegls_encoder_create()
	cdef charls_jpegls_errc charls_jpegls_encoder_set_frame_info(charls_jpegls_encoder *, const charls_frame_info *)
	cdef charls_jpegls_errc charls_jpegls_encoder_set_near_lossless(charls_jpegls_encoder*, int32_t)
	cdef charls_jpegls_errc charls_jpegls_encoder_set_interleave_mode(charls_jpegls_encoder*, const charls_interleave_mode)
	cdef charls_jpegls_errc charls_jpegls_encoder_get_estimated_destination_size(const charls_jpegls_encoder*, size_t*)
	cdef charls_jpegls_errc charls_jpegls_encoder_set_destination_buffer(charls_jpegls_encoder*,void*, size_t)
	cdef charls_jpegls_errc charls_jpegls_encoder_encode_from_buffer(charls_jpegls_encoder*, const void*, size_t, uint32_t)
	cdef charls_jpegls_errc charls_jpegls_encoder_get_bytes_written(const charls_jpegls_encoder*, size_t*)
	cdef void charls_jpegls_encoder_destroy(charls_jpegls_encoder*)


	cdef charls_jpegls_decoder* charls_jpegls_decoder_create();
	cdef charls_jpegls_errc charls_jpegls_decoder_set_source_buffer(charls_jpegls_decoder* decoder, const void* source_buffer, size_t source_size_bytes);
	cdef charls_jpegls_errc charls_jpegls_decoder_read_header(charls_jpegls_decoder* decoder);
	cdef charls_jpegls_errc charls_jpegls_decoder_get_frame_info(const charls_jpegls_decoder* decoder, charls_frame_info* frame_info);
	cdef charls_jpegls_errc charls_jpegls_decoder_get_interleave_mode(const charls_jpegls_decoder* decoder, charls_interleave_mode* interleave_mode);
	cdef charls_jpegls_errc charls_jpegls_decoder_get_destination_size(const charls_jpegls_decoder* decoder, uint32_t stride, size_t* destination_size_bytes);
	cdef charls_jpegls_errc charls_jpegls_decoder_decode_to_buffer(const charls_jpegls_decoder* decoder, void* destination_buffer, size_t destination_size_bytes, uint32_t stride);
	cdef void charls_jpegls_decoder_destroy(const charls_jpegls_decoder* decoder);

def check_charls_failure(error):
	if error != CHARLS_JPEGLS_ERRC_SUCCESS:
		raise ValueError(f'{charls_get_error_message(error)}')
	
def save(tn, f, near_lossless=0):
	cdef charls_jpegls_encoder *encoder = NULL
	cdef charls_frame_info frame_info
	cdef charls_jpegls_errc error
	cdef size_t encoded_buffer_size
	cdef size_t bytes_written = 0
	cdef size_t tn_size = 0

	try:
		tn = np.asarray(tn)

		if len(tn.shape) > 3:
			raise ValueError(f"Unsupported shape={tn.shape} must be 1D, 2D, or 3D!")

		while len(tn.shape) < 3:
			tn = tn.reshape((1, ) + tn.shape)

		if not tn.flags.c_contiguous:
			tn = np.ascontiguousarray(tn)

		encoder = charls_jpegls_encoder_create()
		if encoder == NULL:
			raise MemoryError()

		frame_info.component_count = tn.shape[0]
		frame_info.width = tn.shape[1]
		frame_info.height = tn.shape[2]
		if tn.dtype == np.uint16:
			frame_info.bits_per_sample = 16
			sample_bytes = 2
		elif tn.dtype == np.uint8:
			frame_info.bits_per_sample = 8
			sample_bytes = 1
		else:
			raise ValueError(f"Unsupported dtype={tn.dtype}!")

		tn_size = frame_info.component_count * frame_info.width * frame_info.height * sample_bytes

		error = charls_jpegls_encoder_set_frame_info(encoder, &frame_info)
		check_charls_failure(error)

		error = charls_jpegls_encoder_set_near_lossless(encoder, near_lossless)
		check_charls_failure(error)
		
		error = charls_jpegls_encoder_set_interleave_mode(encoder, CHARLS_INTERLEAVE_MODE_NONE);
		check_charls_failure(error)

		error = charls_jpegls_encoder_get_estimated_destination_size(encoder, &encoded_buffer_size);
		check_charls_failure(error)
		
		encoded_buffer_size = 3 * encoded_buffer_size / 2

		encoded_buffer = np.empty(encoded_buffer_size, dtype=np.uint8)

		error = charls_jpegls_encoder_set_destination_buffer(encoder, <void*>np.PyArray_BYTES(encoded_buffer), encoded_buffer_size)
		check_charls_failure(error)

		error = charls_jpegls_encoder_encode_from_buffer(encoder, np.PyArray_BYTES(tn), tn_size, 0);
		check_charls_failure(error)

		error = charls_jpegls_encoder_get_bytes_written(encoder, &bytes_written);
		check_charls_failure(error)

		f.write(encoded_buffer[:bytes_written])
	finally:
		if encoder != NULL:
			charls_jpegls_encoder_destroy(encoder)

def load(f):
	cdef charls_jpegls_decoder *decoder = NULL
	cdef charls_jpegls_errc error
	cdef charls_frame_info frame_info
	cdef charls_interleave_mode interleave_mode
	cdef size_t destination_size = 0

	try:
		decoder = charls_jpegls_decoder_create()
		if decoder == NULL:
			raise MemoryError()
		
		encoded_buffer = np.frombuffer(f.read(), dtype=np.uint8)
		if not encoded_buffer.flags.c_contiguous:
			encoded_buffer = np.ascontiguousarray(encoded_buffer)
		
		error = charls_jpegls_decoder_set_source_buffer(decoder, np.PyArray_BYTES(encoded_buffer), encoded_buffer.shape[0])
		check_charls_failure(error)

		error = charls_jpegls_decoder_read_header(decoder)
		check_charls_failure(error)

		charls_jpegls_decoder_get_frame_info(decoder, &frame_info)
		check_charls_failure(error)

		error = charls_jpegls_decoder_get_interleave_mode(decoder, &interleave_mode)
		check_charls_failure(error)

		if interleave_mode != CHARLS_INTERLEAVE_MODE_NONE:
			raise ValueError(f"Unsupported interleave_mode={interleave_mode}!")

		error = charls_jpegls_decoder_get_destination_size(decoder, 0, &destination_size)
		check_charls_failure(error)

		shape = (frame_info.component_count, frame_info.width, frame_info.height)
		if frame_info.bits_per_sample == 8:
			tn = np.empty(shape, dtype=np.uint8)
			sample_bytes = 1
		elif frame_info.bits_per_sample == 16:
			tn = np.empty(shape, dtype=np.uint16)
			sample_bytes = 2
		else:
			raise ValueError(f"Unsupported bits_per_sample={frame_info.bits_per_sample}!")

		if not tn.flags.c_contiguous:
			tn = np.ascontiguousarray(tn)

		if destination_size != shape[0] * shape[1] * shape[2] * sample_bytes:
			raise ValueError("Unexpected destination size!")

		error = charls_jpegls_decoder_decode_to_buffer(decoder, np.PyArray_BYTES(tn), destination_size, 0)
		check_charls_failure(error)

		return tn
	finally:
		if decoder != NULL:
			charls_jpegls_decoder_destroy(decoder)
