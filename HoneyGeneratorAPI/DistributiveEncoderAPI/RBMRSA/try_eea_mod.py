# Source: extendedeuclideanalgorithm.com
import math 
import sys
import time
# Added relative importing syntax (.), as it is now a package.
from .try_generating_keys import gen_e

#Warning: can't handle b=0. See extendedeuclideanalgorithm.com/code for a version that can
def gcd_iterative(a, b):
	""" Calculating the greatest common divisor 
	using the Euclidean Algorithm (non-recursive) 
	(Source: extendedeuclideanalgorithm.com/code) 
	"""
  
	#Set default values for the quotient and the remainder
	q = 0  #CONSTANT-- DON'T CHANGE
	r = 1  #CONSTANT-- DON'T CHANGE
	
	'''
	In each iteration of the loop below, we
	calculate the new quotient, remainder, a and b.
	r decreases, so we stop when r = 0 
	'''
	while(r > 0):
		#The calculations
		q = math.floor(a//b)
		r = a - q * b
		
		#The values for the next iteration
		a = b
		b = r if (r > 0) else b
	
	return abs(b)


#Can handle b=0
def gcd_iterative_2(a, b):
   """ Calculating the greatest common divisor 
   using the Euclidean Algorithm (non-recursive) 
   (Source: extendedeuclideanalgorithm.com/code) 
   """
  
   #Set default values for the quotient and the remainder
   q = 0
   r = 1
   
   '''
   In each iteration of the loop below, we
   calculate the new quotient, remainder, a and b.
   r decreases, so we stop when r = 0 
   '''
   while(b > 0):
      #The calculations
      q = math.floor(a//b)
      r = a - q * b
      
      #The values for the next iteration
      a = b
      b = r
   
   return abs(a)


def gcd(a, b):
   """ Calculating the greatest common divisor 
   using the Euclidean Algorithm (recursive) 
   (Source: extendedeuclideanalgorithm.com/code)
   """
   if(b == 0):
      return abs(a)
   
   q = math.floor(a//b)
   r = a - q * b
   return abs(b) if (r == 0) else gcd(b, r)
   
   

#Warning: this version can't handle b=0. See extendedeuclideanalgorithm.com/code for a version that can.
def xgcd_iterative(a, b):
	""" Calculates the gcd and Bezout coefficients, 
	using the Extended Euclidean Algorithm (non-recursive).
	(Source: extendedeuclideanalgorithm.com/code) 
	"""
	#Set default values for the quotient, remainder, 
	#s-variables and t-variables
	q = 0
	r = 1
	s1 = 1 
	s2 = 0
	s3 = 1 
	t1 = 0 
	t2 = 1
	t3 = 0
	
	'''
	In each iteration of the loop below, we
	calculate the new quotient, remainder, a, b,
	and the new s-variables and t-variables.
	r decreases, so we stop when r = 0
	'''
	while(r > 0):
		#The calculations
		q = math.floor(a//b)
		r = a - q * b
		s3 = s1 - q * s2
		t3 = t1 - q * t2
		
		'''
		The values for the next iteration, 
		(but only if there is a next iteration)
		'''
		if(r > 0):
			a = b
			b = r
			s1 = s2
			s2 = s3
			t1 = t2
			t2 = t3

	return abs(b), s2, t2

#Can handle b=0
def xgcd_iterative_2(a, b):
   """ Calculates the gcd and Bezout coefficients, 
   using the Extended Euclidean Algorithm (non-recursive).
   (Source: extendedeuclideanalgorithm.com/code) 
   """
   #Set default values for the quotient, remainder, 
   #s-variables and t-variables
   q = 0
   r = 1
   s1 = 1 
   s2 = 0
   s3 = 1 
   t1 = 0 
   t2 = 1
   t3 = 0
   
   '''
   In each iteration of the loop below, we
   calculate the new quotient, remainder, a, b,
   and the new s-variables and t-variables.
   r decreases, so we stop when r = 0
   '''
   while(b > 0):
      #The calculations
      q = math.floor(a//b)
      r = a - q * b
      s3 = s1 - q * s2
      t3 = t1 - q * t2
      
      '''
      The values for the next iteration, 
      (but only if there is a next iteration)
      '''
     
      a = b
      b = r
      s1 = s2
      s2 = s3
      t1 = t2
      t2 = t3

   return abs(a), s1, t1


def xgcd(a, b, s1 = 1, s2 = 0, t1 = 0, t2 = 1):
   """ Calculates the gcd and Bezout coefficients, 
   using the Extended Euclidean Algorithm (recursive).
   (Source: extendedeuclideanalgorithm.com/code) 
   """
   if(b == 0):
      return abs(a), 1, 0
   
   q = math.floor(a//b)
   r = a - q * b
   s3 = s1 - q * s2
   t3 = t1 - q * t2
   
   #if r==0, then b will be the gcd and s2, t2 the Bezout coefficients
   return (abs(b), s2, t2) if (r == 0) else xgcd(b, r, s2, s3, t2, t3)


def gcd_checker (e, PHI):
    a = e
    b = PHI
    my_gcd, s, t = xgcd(a, b)
    verification = abs(s*a+t*b)
    while my_gcd != verification:
        gen_e()
        a = e
        b = PHI
        my_gcd, s, t = xgcd(a, b)
        verification = abs(s*a+t*b)
    return s, t

def generating_d (x, y, e, PHI):
    a = e
    b = PHI
       
    eea_d = (PHI * x) + (e * y)

    if eea_d == gcd(a, b):
        d = y
        if d > PHI:
            d = d % PHI
        elif d < 0:
            d = d + PHI

        return d
    else:   
        print ("error")

#Use main() as main function when you run this script -- for checking only
if __name__== "__main__":
    e = 19
    PHI = 960
    #final_step(e, PHI)
    print ("--------------------")
    y, x = gcd_checker (e, PHI)
    print (x, y)

    private_key = generating_d (x, y, e, PHI)
    print (private_key)


