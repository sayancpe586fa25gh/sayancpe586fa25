import secrets
import math

# Secure uniform sampler
def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample in [a, b)."""
    u = secrets.randbits(53) / (1 << 53)  # 53-bit precision
    return a + (b - a) * u

def exponentialdist(lmbd: float) -> float:
    #lmbd -> lambda parameter
    u = uniform(0.0, 1.0)
    return -math.log(u) / lmbd

def poissondist(lmbd: float) -> int:
    #lmbd -> lambda parameter 
    L = math.exp(-lmbd)
    k = 0
    p = 1.0
    
    while True:
        u = uniform(0.0, 1.0)
        p *= u
        if p <= L:
            return k
        k += 1
