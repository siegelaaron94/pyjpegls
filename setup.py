from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
	name='pyjpegls',
	version='0.0.1',
	description='Encode and decode jpeg-ls files to and from numpy arrays.',
	install_requires = [
		'numpy'
	],
	packages=find_packages('src'),
	package_dir={'':'src'},
	ext_modules=cythonize(
		Extension('jpegls',
			sources=[
				'src/jpegls.pyx',
				'thirdparty/charls/src/charls_jpegls_decoder.cpp',
				'thirdparty/charls/src/charls_jpegls_encoder.cpp',
				'thirdparty/charls/src/jpegls_error.cpp',
				'thirdparty/charls/src/jpegls.cpp',
				'thirdparty/charls/src/jpeg_stream_reader.cpp',
				'thirdparty/charls/src/jpeg_stream_writer.cpp',
				'thirdparty/charls/src/version.cpp'
			],
			include_dirs=[
				numpy.get_include(),
				'thirdparty/charls/include'
			],
			define_macros =[
				('CHARLS_STATIC', 1),
				('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
			]
		),
		build_dir='build'
	)
)