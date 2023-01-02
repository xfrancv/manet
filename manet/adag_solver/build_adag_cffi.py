import cffi
import pathlib

this_dir = pathlib.Path().resolve()
ffi = cffi.FFI()

#ffi.cdef('float cmult(int int_param, float float_param);'))
ffi.cdef("""
    int adag_maxsum( unsigned char* labels, double *energy,
                      unsigned int nT, unsigned short nK,
                      unsigned short nOmega, unsigned int *Omega, 
                      unsigned short nG, int *G, int *Q, int *f, 
                      unsigned int theta);
""")

ffi.set_source(
    "adag_solver",
    # Since we are calling a fully built library directly no custom source
    # is necessary. We need to include the .h files, though, because behind
    # the scenes cffi generates a .c file which contains a Python-friendly
    # wrapper around each of the functions.
    '#include "libadag.hpp"',
    # The important thing is to include the pre-built lib in the list of
    # libraries we are linking against:
    libraries=["adag"],
    library_dirs=[this_dir.as_posix()],
    extra_link_args=["-Wl,-rpath,."],
)

ffi.compile()
print("* Complete")
