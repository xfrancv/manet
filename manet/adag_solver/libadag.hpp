#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_SYMBOL int adag_maxsum( unsigned char* labels, double *energy,
                                unsigned int nT, unsigned short nK,
                                unsigned short nOmega, unsigned int *Omega, 
                                unsigned short nG, int *G, int *Q, int *f, unsigned int theta);


#ifdef __cplusplus
}
#endif
