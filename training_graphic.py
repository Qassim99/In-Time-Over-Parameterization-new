import os
import argparse
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sys


#test.csv erstellen 
#epochs_own = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#losses_own = [10, 9 , 8, 7, 6, 5, 4, 3, 2, 1]
#savefile = 'test.csv'
#df = pd.DataFrame({'Epochs': epochs_own, 'losses': losses_own})
#df.to_csv(savefile, index=False) 


input_file = str(sys.argv[1])
output_file = str(sys.argv[1].split('.')[0]) + ".pdf"


data = pd.read_csv(input_file)
epochs = data.epochs.to_list()
epochs = [epoch + 1 for epoch in epochs]
#print(data.Epochs.to_list())
plt.figure().set_figheight(2.4)
plt.plot(epochs, data.losses.to_list())
plt.xlabel("Epochenindex")
plt.ylabel("Loss")
plt.grid() 
if(len(epochs) == 10):
    plt.xticks(epochs)
elif(len(epochs) == 70):
    plt.xticks([1, 10, 20, 30, 40, 50, 60, 70])
plt.ylim(ymin=0)
plt.savefig(output_file, format="pdf", bbox_inches="tight")
#plt.show()

#python tests/create_graphics/training_graphic.py cheng2020-anchor_trained_on_Vimeo-90k_sequence7.csv
#python tests/create_graphics/training_graphic.py cheng2020-anchor_trained_on_Vimeo-90k_sequence1.csv