import torch
from fastai.vision.all import *
import matplotlib.pyplot as plt

#add MNIST Dataset
path = untar_data(URLs.MNIST_SAMPLE)

#Check Directory
print(path.ls())
print((path/'train').ls())

#Check labaling 7s and 3s
print("--------------------")
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
print(threes[0:5])
print(f"Number of 3s: {len(threes)}")
print(f"Number of 7s: {len(sevens)}")

#Review one image from the array and save it
im3_path = threes[1]
im3 = Image.open(im3_path)
im3.save('im3.png')

#Represent image as numpy array
im3_array = np.array(im3)
#print(im3_array)

#Represent image as tensor
im3_tensor = tensor(im3)
#print(im3_tensor)

#Represent image with pandas
df = pd.DataFrame(im3_tensor[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
#print(df)

###################################
##
### First Try: Pixel Similarity
##
###################################

#List all threes and sevens numbers
threes_tensors = [tensor(Image.open(o)) for o in threes]
sevens_tensors = [tensor(Image.open(o)) for o in sevens]

print(f"Number of 3 Tensors: {len(threes_tensors)}")
print(f"Number of 7 Tensors: {len(sevens_tensors)}")

#Show some images from the tensor array
show_image(threes_tensors[1])
show_image(sevens_tensors[1])

# Cast values to float
stacked_sevens = torch.stack(sevens_tensors).float()/255
stacked_threes = torch.stack(threes_tensors).float()/255
print(stacked_threes.shape, stacked_sevens.shape)

#check the rank of tensors
print(stacked_threes.ndim, stacked_sevens.ndim)

#Mean all images
mean_3 = stacked_threes.mean(0)
mean_7 = stacked_sevens.mean(0)
i_3 = show_image(mean_3, title='Mean 3')
plt.savefig('mean_3.png')
plt.close()
u_7 = show_image(mean_7, title='Mean 7')
plt.savefig('mean_7.png')
plt.close()

#Diff between ideal and mean
a_3 = stacked_threes[1]
#s = show_image(a_3, title='A 3')
#plt.savefig('a_3.png')
#plt.close()

dist_3_abs = (a_3 -mean_3).abs().mean()
dist_3_sqr = ((a_3 -mean_3)**2).mean().sqrt()
print(f"3 Absolute Distance: {dist_3_abs}")
print(f"3 Squared Distance: {dist_3_sqr}")

dist_7_abs = (a_3 -mean_7).abs().mean()
dist_7_sqr = ((a_3 -mean_7)**2).mean().sqrt()
print(f"7 Absolute Distance: {dist_7_abs}")
print(f"7 Squared Distance: {dist_7_sqr}")
###################################

#instead of using manual function, use fastai library
f1l = F.l1_loss(a_3.float(),mean_7)
fmse = F.mse_loss(a_3,mean_7).sqrt()
print(f"FastAI L1 Loss: {f1l}")
print(f"FastAI MSE Loss: {fmse}")

#Calculate the mean absolute error
valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens = valid_3_tens.float()/255


def mnist_distance(a,b):
    return (a-b).abs().mean((-1,-2))

#print("Mean Abs Error = " , mnist_distance(a_3,mean_3))

def is_3(x):
    return mnist_distance(x,mean_3) < mnist_distance(x,mean_7)

print("Is a 3? ", is_3(a_3))
print("Is a 3? ", is_3(a_3).float())

print(is_3(valid_3_tens))

accuracy_3s = is_3(valid_3_tens).float().mean()
# use logical_not (~) to invert the boolean mask and call mean()
accuracy_7s = (~is_3(valid_7_tens)).float().mean()
print(f"Accuracy on 3s: {accuracy_3s:.4f}")
print(f"Accuracy on 7s: {accuracy_7s:.4f}")
print("Overall Accuracy: ", (accuracy_3s + accuracy_7s)/2)

###################################
##
### Stochastic Gradient Descent
##
###################################

#def pr_eight(x,w) : (x*w).sum()

def f(x): return x**2


#plot_function(f,'f','x**2')

xt = tensor(3.0).requires_grad_()

yt = f(xt)
#print(f"Value of f(3): {yt}")
yt.backward() #compute the gradient
#print(f"Gradient of f at 3: {xt.grad}") #print the gradient 

xt = tensor([3.,4.,10.]).requires_grad_()

def f(x): return (x**2).sum()

yt = f(xt)
#print(f"Value of f(xt): {yt}")
yt.backward() #compute the gradient
#print(f"Gradient of f at xt: {xt.grad}") #print the gradient


###################################
##
### Stepping with a Learning Rate
##
## An End-to-End SGD Example
###################################

time = torch.arange(0.,20).float()
print(time)

speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 +1
plt.scatter(time,speed)
plt.xlabel('Time')
plt.ylabel('Speed')
plt.title('Time vs Speed')
plt.savefig('time_speed.png')
plt.close()


