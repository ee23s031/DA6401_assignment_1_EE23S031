import wandb
import numpy as np
import keras
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

#### Functions ###########

def deri_sigmoid(z):
    return z*(1-z)

def deri_relu(z):
    return np.int64(z>0)


def deri_tanh(z):
    return 1-(Tanh(z)*Tanh(z))



#### Back propagation Algo. Functions ###########
def back_prop(w,b,a,h,ypred,y_hot,x,act,loss_type):
    GRAD_h,GRAD_a,GRAD_w,GRAD_b=[],[],[],[]
    
    if loss_type=='cross_entropy':
        grad_a=-(y_hot-ypred)
        grad_h=0
    elif loss_type=='mean_squared_error':
        grad_a=(ypred-y_hot)*ypred*(1-ypred)
        grad_h=0
    else:
        raise ValueError('Loss function not found')
    
    GRAD_a.append(grad_a)
    GRAD_h.append(grad_h)
    N=x.shape[1]
    
    activations=['sigmoid', 'tanh', 'ReLU','identity']
    if act in activations:
    
        for z,i in enumerate(range(len(h)-1,-1,-1)):


            if z==0:
                GRAD_w.append((GRAD_a[z]@h[i-1].T)/N)
                GRAD_b.append((np.sum(GRAD_a[z],axis=1,keepdims=True))/N)
            elif i!=0:
                grad_h=w[i+1].T@GRAD_a[z-1]
                if act=='sigmoid':
                    grad_a=grad_h*deri_sigmoid(h[i])
                elif act=='ReLU':
                    grad_a=grad_h*deri_relu(h[i])
                elif act=='tanh':
                    grad_a=grad_h*deri_tanh(h[i])
                elif act=='identity':
                    grad_a=grad_h

                GRAD_h.append(grad_h)
                GRAD_a.append(grad_a)

                GRAD_w.append((GRAD_a[z]@h[i-1].T)/N)
                GRAD_b.append((np.sum(GRAD_a[z],axis=1,keepdims=True))/N)
            else:
                grad_h=w[i+1].T@GRAD_a[z-1]
                if act=='sigmoid':
                    grad_a=grad_h*deri_sigmoid(h[i])
                elif act=='ReLU':
                    grad_a=grad_h*deri_relu(h[i])
                elif act=='tanh':
                    grad_a=grad_h*deri_tanh(h[i])
                elif act=='identity':
                    grad_a=grad_h
                GRAD_h.append(grad_h)
                GRAD_a.append(grad_a)

                GRAD_w.append((GRAD_a[z]@x.T)/N)
                GRAD_b.append((np.sum(GRAD_a[z],axis=1,keepdims=True))/N)
                
    else:
        raise ValueError('you have given a wrong activation function')
            
            
    return GRAD_w[::-1],GRAD_b[::-1]



def sgd(grad_w,w,grad_b,b,lr,w_d):
    w_update=[]
    b_update=[]
    for i in range(len(w)):
        w_update.append(w[i]*(1-lr*w_d)-lr*grad_w[i])
        b_update.append(b[i]*(1-lr*w_d)-lr*grad_b[i])
        
    return w_update,b_update
    
            
def momentum(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,beta,w_d):
    if iter_==0:

        w_update=[]
        b_update=[]
        prev_grad_w=[]
        prev_grad_b=[]
        for i in range(len(w)):
            w_update.append(w[i]*(1-lr*w_d)-lr*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-lr*grad_b[i])
            prev_grad_w.append(lr*grad_w[i])
            prev_grad_b.append(lr*grad_b[i])

        return w_update,b_update,prev_grad_w,prev_grad_b
    else:
        w_update=[]
        b_update=[]
        for i in range(len(w)):
            update_w=(lr*grad_w[i]+beta*prev_grad_w[i])
            update_b=(lr*grad_b[i]+beta*prev_grad_b[i])
            w_update.append(w[i]*(1-lr*w_d)-update_w)
            b_update.append(b[i]*(1-lr*w_d)-update_b)
            prev_grad_w[i]=update_w
            prev_grad_b[i]=update_b

        return w_update,b_update,prev_grad_w,prev_grad_b
    


    
def nag(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,beta,w_d,x,y_hot,activation,loss_type):
    if iter_==0:

        w_update=[]
        b_update=[]
        prev_grad_w=[]
        prev_grad_b=[]
        for i in range(len(w)):
            w_update.append(w[i]*(1-lr*w_d)-lr*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-lr*grad_b[i])
            prev_grad_w.append(lr*grad_w[i])
            prev_grad_b.append(lr*grad_b[i])

        return w_update,b_update,prev_grad_w,prev_grad_b
    
    else:
        w_update=[]
        b_update=[]
        w_look=[]
        b_look=[]
        for i in range(len(w)):
            update_w=(beta*prev_grad_w[i])
            update_b=(beta*prev_grad_b[i])
            w_look.append(w[i]*(1-lr*w_d)-update_w)
            b_look.append(b[i]*(1-lr*w_d)-update_b)
        A,H=forward_prop(x,w_look,b_look,act=activation)
        back_prop(w,b,A,H,H[-1],y_hot,x,activation,loss_type)
        grad_w,grad_b=back_prop(w_look,b_look,A,H,H[-1],y_hot,x,activation,loss_type)
        
        for i in range(len(w)):
            prev_grad_w[i]=beta*prev_grad_w[i]+lr*grad_w[i]
            prev_grad_b[i]=beta*prev_grad_b[i]+lr*grad_b[i]
            
            w_update.append(w[i]*(1-lr*w_d)-prev_grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-prev_grad_b[i])

        return w_update,b_update,prev_grad_w,prev_grad_b
        



def rmsprop(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta,w_d):
    
    if iter_==0:
        eps=1e-8

        w_update=[]
        b_update=[]
        vt_w=[]
        vt_b=[]
        for i in range(len(w)):
            vt_w.append((1-beta)*grad_w[i]**2)
            vt_b.append((1-beta)*grad_b[i]**2)
            
            div_w=(1/np.sqrt(vt_w[i]+eps))
            div_b=(1/np.sqrt(vt_b[i]+eps))
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*div_w)*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-(lr*div_b)*grad_b[i])
            

        return w_update,b_update,vt_w,vt_b
    else:
        eps=1e-8
        w_update=[]
        b_update=[]
        
        for i in range(len(w)):
            vt_w[i]=beta*vt_w[i]+(1-beta)*(grad_w[i]**2)
            vt_b[i]=beta*vt_b[i]+(1-beta)*(grad_b[i]**2)
            
            
            div_w=np.multiply(lr,np.reciprocal(np.sqrt(vt_w[i]+eps)))
            div_b=np.multiply(lr,np.reciprocal(np.sqrt(vt_b[i]+eps)))
            
            
            w_update.append(w[i]*(1-lr*w_d)-div_w*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-div_b*grad_b[i])
        return w_update,b_update,vt_w,vt_b
        

    
        
def ADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d):
    if iter_==0:
        eps=1e-8

        w_update=[]
        b_update=[]
        vt_w=[]
        vt_b=[]
        mt_w=[]
        mt_b=[]
        for i in range(len(w)):
            
            vt_w.append((1-beta2)*grad_w[i]**2)
            vt_b.append((1-beta2)*grad_b[i]**2)
            
            mt_w.append((1-beta1)*grad_w[i])
            mt_b.append((1-beta1)*grad_b[i])
            

            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
            

        return w_update,b_update,mt_w,mt_b,vt_w,vt_b
    else:
        eps=1e-8
        w_update=[]
        b_update=[]
        
        for i in range(len(w)):
            

            vt_w[i]=vt_w[i]*beta2+(1-beta2)*grad_w[i]**2
            vt_b[i]=vt_b[i]*beta2+(1-beta2)*grad_b[i]**2
            
            
            mt_w[i]=beta1*mt_w[i]+(1-beta1)*grad_w[i]
            mt_b[i]=beta1*mt_b[i]+(1-beta1)*grad_b[i]
            
        
            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
        
        return w_update,b_update,mt_w,mt_b,vt_w,vt_b
    

    
def NADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d,x,y_hot,activation,loss_type):
    if iter_==0:
        eps=1e-8

        w_update=[]
        b_update=[]
        vt_w=[]
        vt_b=[]
        mt_w=[]
        mt_b=[]
        for i in range(len(w)):
            
            vt_w.append((1-beta2)*grad_w[i]**2)
            vt_b.append((1-beta2)*grad_b[i]**2)
            
            mt_w.append((1-beta1)*grad_w[i])
            mt_b.append((1-beta1)*grad_b[i])
            

            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
            

        return w_update,b_update,mt_w,mt_b,vt_w,vt_b
    else:
        eps=1e-8
        w_update=[]
        b_update=[]
        w_look=[]
        b_look=[]
        
        for i in range(len(w)):
            w_look.append(w[i]-beta1*mt_w[i])
            b_look.append(b[i]-beta1*mt_b[i])
            
        A,H=forward_prop(x,w_look,b_look,act=activation)
        grad_w,grad_b=back_prop(w_look,b_look,A,H,H[-1],y_hot,x,activation,loss_type)
        
        for i in range(len(w)):

            
            mt_w[i]=beta1*mt_w[i]+(1-beta1)*grad_w[i]
            mt_b[i]=beta1*mt_b[i]+(1-beta1)*grad_b[i]
            
            
            
            vt_w[i]=vt_w[i]*beta2+(1-beta2)*grad_w[i]**2
            vt_b[i]=vt_b[i]*beta2+(1-beta2)*grad_b[i]**2
            
            
        
            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
        
        return w_update,b_update,mt_w,mt_b,vt_w,vt_b