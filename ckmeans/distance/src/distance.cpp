#ifdef _WIN32
#define LIBRARY_API extern "C" __declspec(dllexport)
#else
#define LIBRARY_API extern "C"
#endif

#include<iostream>
#include<cstdint>
#include<vector>
#include<math.h>

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

// bases
const std::uint8_t A = 0b10001000; // A
const std::uint8_t G = 0b01001000; // G
const std::uint8_t C = 0b00101000; // C
const std::uint8_t T = 0b00011000; // T
// wobbles
const std::uint8_t R = 0b11000000; // A|G 
const std::uint8_t M = 0b10100000; // A|C 
const std::uint8_t W = 0b10010000; // A|T 
const std::uint8_t S = 0b01100000; // G|C 
const std::uint8_t K = 0b01010000; // G|T 
const std::uint8_t Y = 0b00110000; // C|T 
const std::uint8_t V = 0b11100000; // A|G|C 
const std::uint8_t H = 0b10110000; // A|C|T 
const std::uint8_t D = 0b11010000; // A|G|T 
const std::uint8_t B = 0b01110000; // G|C|T 
const std::uint8_t N = 0b11110000; // A|G|C|T
// extra
const std::uint8_t KNOWN   = 0b00001000; // base is known, i.e. A, G, C, T
const std::uint8_t GAP     = 0b00000100; // gap
const std::uint8_t UNKNOWN = 0b00000010; // base is unknown, e.g. missing data

const std::uint8_t NOT_PURINE     = 0b00110111; // not a unabiguous purine
const std::uint8_t NOT_PYRIMIDINE = 0b11000111; // not a unabiguous pyrimidine

// helper functions
inline bool isA(std::uint8_t base) {return base == A;}
inline bool isG(std::uint8_t base) {return base == G;}
inline bool isC(std::uint8_t base) {return base == C;}
inline bool isT(std::uint8_t base) {return base == T;}

inline bool isKnown(std::uint8_t base) {return (base & KNOWN) == KNOWN;}
inline bool isUnknown(std::uint8_t base) {return base == UNKNOWN;}
inline bool isGap(std::uint8_t base) {return base == GAP;}

inline bool isSameBase(std::uint8_t a, std::uint8_t b) {return (a == b) && isKnown(a);}
inline bool isDifferentBase(std::uint8_t a, std::uint8_t b) {return (a & b) < 0b00010000;}
inline bool isMatch(std::uint8_t a, std::uint8_t b) {return (a == b);}
inline bool isAmbiguousMatch(std::uint8_t a, std::uint8_t b) {return (a & b) > 0b00001111;}

inline bool isPurine(std::uint8_t base) {return (base & NOT_PURINE) == 0;}
inline bool isPyrimidine(std::uint8_t base) {return (base & NOT_PYRIMIDINE) == 0;}
inline bool isTransition(std::uint8_t a, std::uint8_t b) {
    return (isPurine(a) && isPurine(b))
        || (isPyrimidine(a) && isPyrimidine(b));
}
inline bool isTransversion(std::uint8_t a, std::uint8_t b) {return !isTransition(a, b);}

// == distances

// helpers
std::vector<bool> completeDeletionSites(
    std::uint8_t* alignment,
    int n,
    int m
) {
    std::vector<bool> skip(m);
    for (size_t i = 0; i < m; ++i) {
        skip[i] = false;
        for (size_t j = 0; j < n; ++j) {
            std::uint8_t base = alignment[j * m + i];

            // TODO: think about whether it is a good idea to ignore wobbles
            if (isGap(base) || !isKnown(base)) {
                skip[i] = true;
                break;
            }
        }
    }

    return skip;
}

// p-distance
LIBRARY_API void pDistance(
    std::uint8_t* alignment, // nucleotide alignment
    int n,                   // number of entries
    int m,                   // number of sites
    bool pairwiseDeletion,   // gap handling
    double *distMat          // (output) distance matrix
) {
    // pairwise deletion
    if (pairwiseDeletion) {
        for (size_t i_a = 0; i_a < (n - 1); ++i_a) {
            for (size_t i_b = (i_a + 1); i_b < n; ++i_b) {
                // double to avoid casting later
                double nComp = 0;
                double nMatch = 0;
                for (size_t j = 0; j < m; ++j) {
                    std::uint8_t a = alignment[i_a * m + j];
                    std::uint8_t b = alignment[i_b * m + j];

                    // TODO: think about this... This seems to be the same way as in ape
                    // but I'm not sure that it is a good idea to ignore wobbles.
                    if (!(isGap(a) || isGap(b)) && isKnown(a) && isKnown(b)) {
                        nComp += 1;
                        nMatch += isMatch(a, b);
                    }
                }

                double d = 1.0;
                if (nComp > 0) d = 1 - nMatch / nComp;

                distMat[i_a * n + i_b] = d;
                distMat[i_b * n + i_a] = d;
            }
        }
    // complete deletion
    } else {
        // find sites with missing values
        std::vector<bool> skip = completeDeletionSites(alignment, n, m);

        // p distance calculation
        for (size_t i_a = 0; i_a < (n - 1); ++i_a) {
            for (size_t i_b = (i_a + 1); i_b < n; ++i_b) {
                // double to avoid casting later
                double nComp = 0;
                double nMatch = 0;
                for (size_t j = 0; j < m; ++j) {
                    if (skip[j]) continue; // skip if site contains missing value
                    std::uint8_t a = alignment[i_a * m + j];
                    std::uint8_t b = alignment[i_b * m + j];
                    
                    nComp += 1;
                    nMatch += isMatch(a, b);
                }

                double d = 1.0;
                if (nComp > 0) d = 1 - nMatch / nComp;

                distMat[i_a * n + i_b] = d;
                distMat[i_b * n + i_a] = d;
            }
        }
    }
};

// Jukes-Cantor distance
LIBRARY_API void jcDistance(
    std::uint8_t* alignment, // nucleotide alignment
    int n,                   // number of entries
    int m,                   // number of sites
    bool pairwiseDeletion,   // gap handling
    double *distMat          // (output) distance matrix
) {
    // calculate p
    pDistance(
        alignment, n, m,
        pairwiseDeletion,
        distMat
    );

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double d = abs(- (3.0 / 4.0) * log(1 - (4.0 / 3.0) * distMat[i * n + j]));
            if (isnan(d)) d = INFINITY;
            distMat[i * n + j] = d;
        } 
    }
}

// Kimura 2-parameter distance
LIBRARY_API void k2pDistance(
    std::uint8_t* alignment, // nucleotide alignment
    int n,                   // number of entries
    int m,                   // number of sites
    bool pairwiseDeletion,   // gap handling
    double *distMat          // (output) distance matrix
) {
    // pairwise deletion
    if (pairwiseDeletion) {
        for (size_t i_a = 0; i_a < (n - 1); ++i_a) {
            for (size_t i_b = (i_a + 1); i_b < n; ++i_b) {
                // double to avoid casting later
                double nComp = 0;
                double nTransitions = 0;
                double nTransversions = 0;
                for (size_t j = 0; j < m; ++j) {
                    std::uint8_t a = alignment[i_a * m + j];
                    std::uint8_t b = alignment[i_b * m + j];

                    // TODO: think about this... This seems to be the same way as in ape
                    // but I'm not sure that it is a good idea to ignore wobbles.
                    if (!(isGap(a) || isGap(b)) && isKnown(a) && isKnown(b)) {
                        nComp += 1;
                        // if bases are the same there is neither transition
                        // not transversion
                        if (isMatch(a, b)) continue;

                        bool isTs = isTransition(a, b);
                        nTransitions += isTs;
                        nTransversions += 1 - isTs;
                    }
                }

                double d = INFINITY;
                if (nComp > 0) {
                    double p = nTransitions / nComp;
                    double q = nTransversions / nComp;

                    d = abs(-(1.0 / 2.0) * log((1 - 2 * p - q) * sqrt(1 - 2 * q)));
                    if (isnan(d)) d = INFINITY;
                } 

                distMat[i_a * n + i_b] = d;
                distMat[i_b * n + i_a] = d;
            }
        }
    // complete deletion
    } else {
        // find sites with missing values
        std::vector<bool> skip = completeDeletionSites(alignment, n, m);

        for (size_t i_a = 0; i_a < (n - 1); ++i_a) {
            for (size_t i_b = (i_a + 1); i_b < n; ++i_b) {
                // double to avoid casting later
                double nComp = 0;
                double nTransitions = 0;
                double nTransversions = 0;
                for (size_t j = 0; j < m; ++j) {
                    if (skip[j]) continue; // skip if site contains missing value
                    std::uint8_t a = alignment[i_a * m + j];
                    std::uint8_t b = alignment[i_b * m + j];

                    nComp += 1;
                    // if bases are the same there is neither transition
                    // not transversion
                    if (isMatch(a, b)) continue;

                    bool isTs = isTransition(a, b);
                    nTransitions += isTs;
                    nTransversions += 1 - isTs;
                }

                double d = INFINITY;
                if (nComp > 0) {
                    double p = nTransitions / nComp;
                    double q = nTransversions / nComp;

                    d = abs(-(1.0 / 2.0) * log((1 - 2 * p - q) * sqrt(1 - 2 * q)));
                    if (isnan(d)) d = INFINITY;
                } 

                distMat[i_a * n + i_b] = d;
                distMat[i_b * n + i_a] = d;
            }
        }
    }
}
