import numpy as np
import skimage as ski
from skimage import io, transform, color
import matplotlib.pyplot as plt
import cvxpy as cvx
import copy

def axis_set(L,shape):
    """generate a set of L axes surrounding the image, returning R-- the length of each axis, 
    the set of axis origins relative to the image center, normals, and tangents-- the vectors
    characterising each axis. L must be odd."""
    l,w = shape 
    R = np.sqrt(l**2+w**2)/2
    angles = np.arange(0,2*np.pi,2*np.pi/L)
    origins = [np.array([R*np.cos(a)+l/2,R*np.sin(a)+w/2]) for a in angles]
    normals = [np.array([np.cos(a),np.sin(a)]) for a in angles] #directions from center of image to origins of axes
    tangents = [np.cross(np.array([*n,0]),np.array([0,0,-1.0]))[:-1] for n in normals] #directions of axes    
    return origins,normals,tangents

def project(points,shape,L,M):
    """project a set of points into the coordinate frames surrounding the image as defined in axis_set. 
    Output is list of L projected (but not compressed) vectors F-- one for each frame"""
    shift = np.array(shape)/2#to shift vectors to be from the origin
    origins,normals,tangents = axis_set(L,shape) #size of axes and their parameters
    m = M//2 #center of compressed vector 
    F = []
    for o,n,t in zip(origins,normals,tangents): #for each axis direction
        f = np.zeros(shape=M,dtype=float) #vector representation of points in this frame
        for pt in points: 
            #first shift the point to be from the origin o 
            pt = np.array(pt)-o # p seen from the origin o 
            #print(np.dot(pt,t))
            proj_t = np.round(m+np.dot(pt,t)).astype('int')# the projection of p onto t 
            proj_n = np.dot(pt,n) # the projection of p onto n
            f[proj_t]=proj_n 
        
        F.append(f)
    return F          


def sensing_matrix(N,M):
    """the sensing matrix. N is the length of the compressed representation Y
    M is the length of the uncompressed representation F, which should be 
    int(sqrt(l**2+w**2)) where (l,w) = image.shape"""
    return np.random.uniform(size=(N,M))

def encode(F,N,S):
    """compress a vector representation of length M>N to length N with gauss random sensing"""
    M, = F[0].shape
    Y = []
    for f in F: 
        Y.append(np.matmul(S,f))
    return Y #return list of compressed vectors and the sensing matrix 

def decode(Y,S):
    """decode compressed vectors Y into the expanded representations F
    Y = [y1,y2,...] is a list of all compressed vectors to be decoded
    output F_hat """
    N,M = S.shape
    F_hat = []
    for y in Y: 
        f = cvx.Variable(M)
        objective = cvx.Minimize(cvx.norm(f,1))
        constraints = [S*f == y]
        prob = cvx.Problem(objective,constraints)
        result = prob.solve(verbose=False)
        f = np.array(f.value)
        #f = np.array([a for b in f for a in b])
        #f[np.abs(f)<1e-9]=0
        F_hat.append(f)
    return np.array(F_hat).squeeze() #return prediction of un-compressed vector representations of Y vectors 


def unproject(F_hat,shape,L,M): 
    """given predictions of pre-compression vectors, unproject them back to point predictions"""
    l,w = shape
    shift = np.array([l/2,w/2])#to shift vectors to be from the origin
    origins,normals,tangents = axis_set(L,shape) #size of axes and their parameters
    m = M//2 #center of compressed vector 
    points = []
    for i,f_hat in enumerate(F_hat): 
        o,n,t = origins[i],normals[i],tangents[i]
        inds, = np.nonzero(f_hat)
        for j in inds: 
            proj_n = f_hat[j]
            proj_t = j-m 
            pt = t*proj_t+ n*proj_n # this is from the origin of the coordinate frame
            # now go from the origin of the coordinate frame to the corner of the image 
            pt = pt + o# + shift
            points.append(pt)
    return np.array(points)
            
def draw_plus(p,im,col=[1.0,0,1.0]):
    s0,s1 = im.shape[:2]
    if len(im.shape)<3:
        im2 = np.stack((im,)*3,-1)
    else:
        im2 = im 
    x,y = p
    w = 2
    a = p-np.array([w,0])
    b = p+np.array([w,0])
    c = p-np.array([0,w])
    d = p+np.array([0,w])
    a,b,c,d = np.array([a,b,c,d],dtype=int)
    #if np.alltrue(np.array([ 0<x[0]<l and 0<x[1]<w for x in [a,b,c,d]])):
    if 0<a[0]<s0 and 0<a[1]<s1 and 0<b[0]<s0 and 0<b[1]<s1:
        im2[ski.draw.line(*a,*b)]=col
    if 0<c[0]<s0 and 0<c[1]<s1 and 0<d[0]<s0 and 0<d[1]<s1:
        im2[ski.draw.line(*c,*d)]=col
    return im2        
        
        
def view_points(im,points):
    im2 = np.stack((im,)*3,-1)
    l,w = im.shape
    for p in points:
        if 0<p[0]<l and 0<p[1]<w:
            im2 = draw_plus(p,im2)
        else:
            #print('prediction outside image')
            pass
    plt.imshow(im2)   
    
def decompress(Y,S,shape,L,M):
    """ decode and unproject the encoded points signal Y given sensing matrix S, 
    shape of image shape, length of compressed representation L, and length of 
    uncompressed representation M = int(round(sqrt(shape[0]**2+shape[1]**2)))"""
    return unproject(decode(Y,S),shape,L,M)