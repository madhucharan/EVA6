# Backpropogation

![formula](https://render.githubusercontent.com/render/math?math=e^{i%20%\pi}%20%=%20%-1)

In this assignment, we have made use of the Gradient Descend to train a Neural Network.
In the image, we have the following main components:

IMAGE SCREENSHOT OF NETWORK
- Inputs ( i1 and i2)
- Weights (w1, w2,w3,w4,w5,w6,w7,w8)
- Hidden Neurons (h1, h2)
- Output Neurons (o1, o2)
- Activated Neurons (out_h1, out_h2, out_o1, out_o2)
- Error because of outputs (E1, E2)
- Total Error (E_total)
- There's no bias!

## Forward Propagation

- We start with our inital inputs as i1 = 0.05 and i2 = 0.1
- The outputs we want to acheive are t1 = 0.01 and t2 = 0.99 
- The initial weights are w1 = 0.15 , w2 = 0.2 , w3 = 0.25 , w4 = 0.3
- We perform a FORWARD PASS through the network
- The h1 is calulated by: $$h_{1} = i_{1}\times w_{1}+i_{2}\times w_{2}$$ 
- Similarly, h2 is calulated by: $$h_{2} = i_{1}\times w_{3}+i_{2}\times w_{4}$$
- The output of these hidden neurons are given after performing Sigmoid Activation:
- Sigmoid is given as :  $$\sigma(h)=(1 / 1 + \exp(h))$$
- Thus, a_h1 is given by: $$a_{h1}=\sigma({h_1})$$
- And similarly, $$a_{h2}=\sigma({h_2})$$
- Using the same rules, we get outputs as:
- $$o_1=w_5a_{h_1}+w_6a_{h_2} \ \ and \ \ o_2=w_7a_{h_1}+w_8a_{h_2}$$
- After applying sigmoid: $$a_{o_1}=\sigma(o_1)\ \ and \ \ a_{o_2}=\sigma(o_2)$$

## Error Calculation
- The errors are due to the differences in the expected outputs from the neuron and the received outputs
- The output from the output neurons were $${o_1} \ and \ {o_2}$$, whereas the expected outputs (i.e., the target values) are $${t_1} \ and \ {t_2}$$
- The error is quantified by the squared error of their difference: 
- $$E_1=(t_1-a_{o_1})^2 \ .\ E_2=(t_2-a_{o_1})^2$$
- And total error is $$E_{total}=E_1 \ +\ E_2$$

## Weight updates (Back Propogation)
- To update weights, we use Gradient Descent:
$$W_{new}=W_{old}-\eta\nabla{E_{total}}$$
- Here, $$W_{new}$$ is the updated weight after performing a backward pass
$$W_{old}$$ is the old weight, $$\eta$$ is the learing rate and $$\nabla{E_{total}}$$ is gradient of the total loss
- To compute these derivatives, we use partial derivatives: 
      - We start from $$E_{total}$$ and calculate partial derviative with respect to $$w_5$$:
      - $$E_{1}$$ is the part of the Error whcih is caused because of $$w_5$$ 
          $$\frac{\partial E_1}{\partial w_5} = \frac{\partial E_1}{\partial a_{o_1}} \frac{\partial a_{o_1}}{\partial o_1} \frac{\partial o_1}{\partial w_5}$$ (using Chain Rule )
- From this equation, we can calculate $$\frac{\partial E_1}{\partial a_{o_1}}$$ as $$a_{o_1}-t_1$$
- And, $$\frac{\partial a_{o_1}}{\partial o_1} = (a_{o_1})(1-a_{o_1})$$
- Finally, $$\frac{\partial o_1}{\partial w_5} = a-o_1$$
- Using these 4 equations, we come up with the weight update formula:
 $$W_{5new} = W_{5old} - \eta \frac{\partial E_1}{\partial w_5}$$
- Similarly we can calculate weight updtaes for $$w_6$$, $$w_7$$ and $$w_8$$.

- Coming to the 1st layer, the weights $$w_1$$,$$w_2$$, $$w_3$$ and $$w_4$$ are updated using the formula:
- $$\frac{\partial E_t}{\partial w_1} = \frac{\partial E_t}{\partial a_{o_1}} \frac{\partial a_{o_1}}{\partial o_1 } \frac{\partial o_1}{\partial a_{h_1}} \frac{\partial a_{h_1}}{\partial h_1} \frac{\partial{h_1}}{\partial{w_1}}$$ and similarly for others
- Once all the weights are updated then once again the cycle of forward propagtion and backward propagation repeats as many times as the number of epochs is.
## Learing Rate

- The rate at which updation of weights takes place is controlled by the learning rate $$\eta$$
- Learning Rate $$\eta = 0.1$$ :
- GRAPH
- Learning Rate $$\eta = 0.2$$ :
- GRAPH
- Learning Rate $$\eta = 0.5$$ :
- GRAPH
- Learning Rate $$\eta = 0.8$$ :
- GRAPH
- Learning Rate $$\eta = 1$$ :
- GRAPH
- Learning Rate $$\eta = 2$$ :
- GRAPH
