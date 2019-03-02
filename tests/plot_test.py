import unittest
import random
import numpy as np
import plot
class TestPlot(unittest.TestCase):

    def test_smooth(self):
        # load data
        orig_dat = np.loadtxt('original.txt',delimiter = ',') 
        smooth_dat = np.loadtxt('smooth_data.txt',delimiter = ',') 
        orig_data = np.split(orig_dat,5)
        orig_data = np.array(orig_data)
        smooth_data = np.split(smooth_dat,5)
        smooth_data = np.array(smooth_data)

        # test 5 different data sets
        for i in range(5):
            xs = np.array(orig_data)[i,:,0]
            ys = np.array(orig_data)[i,:,1]
            x_after, y_after,_ = plot.smooth(xs,ys,decay_steps=3)
            self.assertListEqual(x_after.tolist(), smooth_data[i,:,0].tolist())
            self.assertListEqual(y_after.tolist(), smooth_data[i,:,1].tolist())
            
            
            

if __name__ == "__main__":
    unittest.main()