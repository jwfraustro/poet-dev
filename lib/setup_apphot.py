from distutils.core import setup, Extension

setup(name="apphot_c", version="1.0",
	ext_modules = [Extension("apphot_c", ["apphot_c.c"])])
