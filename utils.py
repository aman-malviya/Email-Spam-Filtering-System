#Classes for minimal repition of code

import matplotlib.pyplot as plt
import numpy as np

#Plotting Figures
class Figure:
    def __init__(self,a,b) -> None:
        self.rows=a
        self.cols=b
        self.fig=plt.figure()

    def plot_for_me(self,index,image,title=""):
        self.fig.add_subplot(self.rows,self.cols,index)
        plt.title(title)
        if image != [[]]:
            plt.imshow(image, cmap="gray")
            plt.axis("off")
    
    def add(self,shape):
        self.fig.axes[0].add_artist(shape)
        

#Transformations
class TF:
    def __init__(self) -> None:
        pass

    def log(image, constant=1):
        return constant*np.log(1+abs(image))

    def normalize(image):
        image=abs(image)
        image=image-image.min()
        image=255*image/image.max()
        return image