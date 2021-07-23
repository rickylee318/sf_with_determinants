import numpy as np
import cupy as cp
from cupyx.scipy.special import ndtr
from cupy.core import internal


TRUNCNORM_TAIL_X = 30
TRUNCNORM_MAX_BRENT_ITERS = 40

s2pi = 2.50662827463100050242E0
P0 = cp.array([-5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0])
Q0 = cp.array([1.00000000000000000000E0,
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0])
    
P1 = cp.array([4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4])
    
Q1 = cp.array([1.00000000000000000000E0,
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4])

P2 = cp.array([3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9])

Q2 = cp.array([1.00000000000000000000E0,
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9])

class TruncNormal():
    """Normal(mu, sigma^2) truncated to [a, b] interval.
    """
    def __init__(self, a=None, b=None, loc=None, scale=None, size=None):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        self.size = size

    def norm_logcdf(self, x):
        return cp.log(ndtr(x))

    def norm_logsf(self, x):
        return self.norm_logcdf(-x)

    def norm_cdf(self,x):
        return ndtr(x)

    def norm_sf(self,x):
        return norm_cdf(-x)
    def norm_isf(self,q):
        return -self.ndtri(q)

    def ndtri(self,y0):
        out = cp.zeros(y0.shape)
        cond1 = y0 == 0.0
        cond2 = y0 == 1.0
        if cp.any(cond1) == True:
            y0[cond1] = -cp.inf
        if cp.any(cond2) == True:
            y0[cond2] = cp.inf
        code = cp.ones(y0.shape, dtype=bool)
        cond3 = y0 > (1.0 - 0.13533528323661269189)
        if cp.any(cond3) == True:
            y0[cond3] = 1.0 - y0[cond3]
            code[cond3] = 0
        cond4 = y0 > 0.13533528323661269189
        cond5 = y0 <= 0.13533528323661269189
        x = cp.sqrt(-2.0 * cp.log(y0))
        cond6 = (x < 8.0) & cond5
        cond7 = (x >= 8.0) & cond5
        x0 = x - cp.log(x) / x
        z = 1.0 / x
        if cp.any(cond6) == True:
            x1 = x0[cond6] - z[cond6] * cp.polyval(P1,z[cond6]) / cp.polyval(Q1, z[cond6])
            out[cond6] = x1
        if cp.any(cond7) == True:
            x2 = x0[cond7] - z[cond7] * cp.polyval(P2, z[cond7]) / cp.polyval(Q2, z[cond7])
            out[cond7] = x2
        out[code] = -out[code]
        if cp.any(cond4) == True:
            y = y0[cond4]
            y = y - 0.5
            y2 = y * y
            x = y + y * (y2 * cp.polyval(P0,y2) / cp.polyval(Q0, y2))
            x = x * s2pi
            out[cond4] = x
        return out

    def _norm_ilogcdf(self,y):
        """Inverse function to _norm_logcdf==sc.log_ndtr."""
        # Apply approximate Newton-Raphson
        # Only use for very negative values of y.
        # At minimum requires y <= -(log(2pi)+2^2)/2 ~= -2.9
        # Much better convergence for y <= -10
        z = -cp.sqrt(-2 * (y + cp.log(2*cp.pi)/2))
        for _ in range(4):
            z = z - (self.norm_logcdf(z) - y) / self._norm_logcdfprime(z)
        return z
        
    def _norm_logcdfprime(self,z):
        # derivative of special.log_ndtr (See special/cephes/ndtr.c)
        # Differentiate formula for log Phi(z)_truncnorm_ppf
        # log Phi(z) = -z^2/2 - log(-z) - log(2pi)/2
        #              + log(1 + sum (-1)^n (2n-1)!! / z^(2n))
        # Convergence of series is slow for |z| < 10, but can use
        #     d(log Phi(z))/dz = dPhi(z)/dz / Phi(z)
        # Just take the first 10 terms because that is sufficient for use
        # in _norm_ilogcdf
        #assert cp.all(z <= -10)
        lhs = -z - 1/z
        denom_cons = 1/z**2
        numerator = 1
        pwr = cp.ones(z.shape)
        denom_total, numerator_total = cp.zeros(z.shape), cp.zeros(z.shape)
        sign = -1
        for i in range(1, 11):
            pwr *= denom_cons
            numerator *= 2 * i - 1
            term = sign * numerator * pwr
            denom_total += term
            numerator_total += term * (2 * i) / z
            sign = -sign
        return lhs - numerator_total / (1 + denom_total)
    
    def _truncnorm_get_delta(self,N): #get N delta
        delta = cp.zeros([N,])
        cp.place(delta,(self.a > TRUNCNORM_TAIL_X) & (self.b < -TRUNCNORM_TAIL_X),0)
        delta[self.a>0] = self.norm_sf(self.a[self.a>0])
        delta[self.a<=0] = 1 - self.norm_cdf(self.a[self.a<=0])
        delta[delta<0] = 0
        return delta

    def _truncnorm_ppf(self,q, N):
        out = cp.zeros(cp.shape(q))
        delta = self._truncnorm_get_delta(N)
        cond1 = delta > 0
        cond2 = (delta > 0) & (self.a > 0)
        cond21 = (delta > 0) & (self.a<=0)
        if cp.any(cond1) == True:
            sa = self.norm_sf(a[cond2])
            out[:,cond2] = -self.ndtri((1 - q[:,cond2]) * sa)
        if cp.any(cond21) == True:
            na = norm_cdf(self.a[cond21])
            out[:,cond21] = self._ndtri(q[:,cond21]  + na * (1.0 - q[:,cond21]))
        cond3 = ~cond1 & cp.isinf(self.b)
        cond4 = ~cond1 & cp.isinf(self.a)
        if cp.any(cond3) == True:
            out[:,cond3] = -self._norm_ilogcdf(cp.log1p(-q[:,cond3]) + self.norm_logsf(self.a[cond3]))
        if cp.any(cond4) == True:
            out[:,cond4] = self._norm_ilogcdf(cp.log(q) + self.norm_logcdf(self.b))
        cond5 = out < self.a
        if cp.any(cond5) == True:
            out[cond5] = ((cond5) * self.a)[cond5]
        return out

    def rvs(self, a, b, loc, scale, size):
        self.a, self.b = cp.broadcast_arrays(a, b)
        H,N = size
        U = cp.random.uniform(low=0, high=1, size=(H,N))
        x = self._truncnorm_ppf(U, N)
        x = x*scale + loc
        return x