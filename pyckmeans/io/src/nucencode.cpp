#ifdef _WIN32
#define LIBRARY_API extern "C" __declspec(dllexport)
#else
#define LIBRARY_API extern "C"
#endif

#include<cstdint>
#include<cstddef>

/*
 *    Base encoding as used by R package ape.
 *    See http://ape-package.ird.fr/misc/BitLevelCodingScheme.html
 *    
 *    Summary:
 *    Most significant four bits are base information (A, G, C, T)
 *      76543210       
 *    0b00001000 -> base is known
 *    0b00000100 -> gap
 *    0b00000010 -> unknown base
 * 
 *    bases
 *    A     0b10001000
 *    G     0b01001000
 *    C     0b00101000
 *    T     0b00011000
 * 
 *    wobbles
 *    R     0b11000000      A|G 
 *    M     0b10100000      A|C 
 *    W     0b10010000      A|T 
 *    S     0b01100000      G|C 
 *    K     0b01010000      G|T 
 *    Y     0b00110000      C|T 
 *    V     0b11100000      A|G|C 
 *    H     0b10110000      A|C|T 
 *    D     0b11010000      A|G|T 
 *    B     0b01110000      G|C|T 
 *    N     0b11110000      A|G|C|T 
 *    
 *    gap
 *    -     0b00000100
 * 
 *    unknown/missing state
 *    ?     0b00000010
 *
*/

// ASCII code to nucleotide encoding map
const std::uint8_t asciiToEncoding[128] = {
//  0    1    2    3    4    5    6    7    8    9
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // 000
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // 010
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // 020
    0,   0,   4,   0,   0,   0,   0,   0,   0,   0, // 030
    0,   0,   0,   0,   0,   4,   0,   0,   0,   0, // 040
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // 050
    0,   0,   0,   2,   0, 136, 112,  40, 208,   0, // 060
    0,  72, 176,   0,   0,  80,   0, 160, 240,   0, // 070
    0,   0, 192,  96,  24,   0, 224, 144,   0,  48, // 080
    0,   0,   0,   0,   0,   0,   0, 136, 112,  40, // 090
    208, 0,   0,  72, 176,   0,   0,  80,   0, 160, // 100
    240, 0,   0,   0, 192,  96,  24,   0, 224, 144, // 110
    0,  48,   0,   0,   0,   0,   4,   0            // 120
};


// encode nucleotides in place
LIBRARY_API void encodeNucleotides(
    std::uint8_t* alignment, // nucleotide alignment
    int n,                   // number of entries
    int m                    // number of sites
) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            alignment[i * m + j] =  asciiToEncoding[alignment[i * m + j]];
        }
    }
}

LIBRARY_API void encodeNucleotides_uint32(
    std::uint32_t* alignment, // nucleotide alignment
    int n,                    // number of entries
    int m,                    // number of sites
    std::uint8_t* alignmentEncoded
) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            alignmentEncoded[i * m + j] =  asciiToEncoding[alignment[i * m + j]];
        }
    }
}
