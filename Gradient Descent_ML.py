#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[45]:


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    learning_rate = 0.01
    n = len(x)
    
# Plot the original data points
    plt.scatter(x, y, color='red', marker='+', linewidth=5)
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)]) 
        
        # Plot the prediction line every 500 iterations
        if i % 500 == 0:
            plt.plot(x, y_predicted, color='green', alpha=0.5)  # Plotting with lower opacity to see progression
              
        
    # Gradient calculations
        
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        
    # updated m and b   
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iterations {}".format(m_curr, b_curr, cost, i))
        
  # Final plot adjustments
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient Descent Line Fitting")
    plt.show()

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
        
gradient_descent(x, y)
        


# In[39]:


import matplotlib.pyplot as plt
import numpy as np

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    learning_rate = 0.0001
    n = len(x)
    
    # Scatter plot for original data points
    plt.scatter(x, y, color='red', marker='+', linewidth=5)
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        
        # Plot the line every 500 iterations
        if i % 500 == 0:
            plt.plot(x, y_predicted, color='green')
        
        # Gradient calculations
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        
        # Update m and b
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
    
    # Final plot adjustments
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient Descent Line Fitting")
    plt.show()

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Run the gradient descent function
gradient_descent(x, y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




