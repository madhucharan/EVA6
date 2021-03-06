# Backpropogation


In this assignment, we have made use of the Gradient Descend to train a Neural Network.
In the image, we have the following main components:

[![Screen-Shot-2021-05-28-at-10-37-20-PM.png](https://i.postimg.cc/mrvJsjvf/Screen-Shot-2021-05-28-at-10-37-20-PM.png)](https://postimg.cc/hJ0p8L4p)

- Inputs ( ![formula](https://render.githubusercontent.com/render/math?math=i_1) and ![formula](https://render.githubusercontent.com/render/math?math=i_2))
- Weights (![formula](https://render.githubusercontent.com/render/math?math=w_1),![formula](https://render.githubusercontent.com/render/math?math=w_2),![formula](https://render.githubusercontent.com/render/math?math=w_3),![formula](https://render.githubusercontent.com/render/math?math=w_4),![formula](https://render.githubusercontent.com/render/math?math=w_5),![formula](https://render.githubusercontent.com/render/math?math=w_6),![formula](https://render.githubusercontent.com/render/math?math=w_7),![formula](https://render.githubusercontent.com/render/math?math=w_8))
- Hidden Neurons (![formula](https://render.githubusercontent.com/render/math?math=h_1),![formula](https://render.githubusercontent.com/render/math?math=h_2))
- Output Neurons (![formula](https://render.githubusercontent.com/render/math?math=o_1), ![formula](https://render.githubusercontent.com/render/math?math=o_2))
- Activated Neurons (![formula](https://render.githubusercontent.com/render/math?math=out_{h1}),![formula](https://render.githubusercontent.com/render/math?math=out_{h2}),![formula](https://render.githubusercontent.com/render/math?math=out_{o1}),![formula](https://render.githubusercontent.com/render/math?math=out_{o2}))
- Error because of outputs (![formula](https://render.githubusercontent.com/render/math?math=E_{1}),![formula](https://render.githubusercontent.com/render/math?math=E_{2}))
- Total Error (![formula](https://render.githubusercontent.com/render/math?math=E_{total}))
- There's no bias!

## Forward Propagation

- We start with our inital inputs as ![formula](https://render.githubusercontent.com/render/math?math=i_1%20=%200.05) and ![formula](https://render.githubusercontent.com/render/math?math=i_2%20=%200.1)
- The outputs we want to acheive are ![formula](https://render.githubusercontent.com/render/math?math=t_1%20=%200.01) and ![formula](https://render.githubusercontent.com/render/math?math=t_2%20=%200.99) 
- The initial weights are ![formula](https://render.githubusercontent.com/render/math?math=w_1%20=%200.15) ,![formula](https://render.githubusercontent.com/render/math?math=w_2%20=%200.2) , ![formula](https://render.githubusercontent.com/render/math?math=w_3%20=%200.25) , ![formula](https://render.githubusercontent.com/render/math?math=w_4%20=%200.3)
- We perform a FORWARD PASS through the network
- The h1 is calulated by: ![formula](https://render.githubusercontent.com/render/math?math=h_{1}%20=%20i_{1}\times%20w_{1}%2Bi_{2}\times%20w_{2})
- Similarly, h2 is calulated by: ![formula](https://render.githubusercontent.com/render/math?math=h_{2}%20=%20i_{1}\times%20w_{3}%2Bi_{2}\times%20w_{4})
- The output of these hidden neurons are given after performing Sigmoid Activation:
- Sigmoid is given as :  ![formula](https://render.githubusercontent.com/render/math?math=\sigma(h)=(1%20/%201%20%2B%20\exp(h)))
- Thus, a_h1 is given by: ![formula](https://render.githubusercontent.com/render/math?math=a_{h1}=\sigma({h_1}))
- And similarly, ![formula](https://render.githubusercontent.com/render/math?math=a_{h2}=\sigma({h_2}))
- Using the same rules, we get outputs as:
- ![formula](https://render.githubusercontent.com/render/math?math=o_1=w_5a_{h_1}%2Bw_6a_{h_2}%20\%20\%20and%20\%20\%20o_2=w_7a_{h_1}%2Bw_8a_{h_2})
- After applying sigmoid: ![formula](https://render.githubusercontent.com/render/math?math=a_{o_1}=\sigma(o_1)\%20\%20and%20\%20\%20a_{o_2}=\sigma(o_2))

## Error Calculation
- The errors are due to the differences in the expected outputs from the neuron and the received outputs
- The output from the output neurons were ![formula](https://render.githubusercontent.com/render/math?math={o_1}%20\%20and%20\%20{o_2}), whereas the expected outputs (i.e., the target values) are ![formula](https://render.githubusercontent.com/render/math?math={t_1}%20\%20and%20\%20{t_2})
- The error is quantified by the squared error of their difference: 
- ![formula](https://render.githubusercontent.com/render/math?math=E_1=(t_1-a_{o_1})^2%20\%20.\%20E_2=(t_2-a_{o_1})^2)
- And total error is ![formula](https://render.githubusercontent.com/render/math?math=E_{total}=E_1%20\%20%2B\%20E_2)

## Weight updates (Back Propogation)
- To update weights, we use Gradient Descent:
![formula](https://render.githubusercontent.com/render/math?math=W_{new}=W_{old}-\eta\nabla{E_{total}})
- Here, ![formula](https://render.githubusercontent.com/render/math?math=W_{new}) is the updated weight after performing a backward pass
![formula](https://render.githubusercontent.com/render/math?math=W_{old}) is the old weight, ![formula](https://render.githubusercontent.com/render/math?math=\eta) is the learing rate and ![formula](https://render.githubusercontent.com/render/math?math=\nabla{E_{total}}) is gradient of the total loss
- To compute these derivatives, we use partial derivatives: 
      - We start from ![formula](https://render.githubusercontent.com/render/math?math=E_{total}) and calculate partial derviative with respect to ![formula](https://render.githubusercontent.com/render/math?math=w_5):
      - ![formula](https://render.githubusercontent.com/render/math?math=E_{1}) is the part of the Error whcih is caused because of ![formula](https://render.githubusercontent.com/render/math?math=w_5) 
          ![formula](https://render.githubusercontent.com/render/math?math=\frac{\partial%20E_1}{\partial%20w_5}%20=%20\frac{\partial%20E_1}{\partial%20a_{o_1}}%20\frac{\partial%20a_{o_1}}{\partial%20o_1}%20\frac{\partial%20o_1}{\partial%20w_5}) (using Chain Rule )
- From this equation, we can calculate ![formula](https://render.githubusercontent.com/render/math?math=\frac{\partial%20E_1}{\partial%20a_{o_1}}%20) as ![formula](https://render.githubusercontent.com/render/math?math=a_{o_1}-t_1)
- And, ![formula](https://render.githubusercontent.com/render/math?math=\frac{\partial%20a_{o_1}}{\partial%20o_1}%20=%20(a_{o_1})(1-a_{o_1}))
- Finally, ![formula](https://render.githubusercontent.com/render/math?math=\frac{\partial%20o_1}{\partial%20w_5}%20=%20a-o_1)
- Using these 4 equations, we come up with the weight update formula:
 ![formula](https://render.githubusercontent.com/render/math?math=W_{5new}%20=%20W_{5old}%20-%20\eta%20\frac{\partial%20E_1}{\partial%20w_5})
- Similarly we calculated weight updtaes for ![formula](https://render.githubusercontent.com/render/math?math=w_6), ![formula](https://render.githubusercontent.com/render/math?math=w_7) and ![formula](https://render.githubusercontent.com/render/math?math=w_8) for the purpose of Back Propogation.

- Back Propogation goes from Outputs to the 1st layer, updating weights in its way. 
- Coming to the 1st layer, the weights ![formula](https://render.githubusercontent.com/render/math?math=w_1),![formula](https://render.githubusercontent.com/render/math?math=w_2), ![formula](https://render.githubusercontent.com/render/math?math=w_3) and ![formula](https://render.githubusercontent.com/render/math?math=w_4) are updated using the formula:
- ![formula](https://render.githubusercontent.com/render/math?math=\frac{\partial%20E_t}{\partial%20w_1}%20=%20\frac{\partial%20E_t}{\partial%20a_{o_1}}%20\frac{\partial%20a_{o_1}}{\partial%20o_1%20}%20\frac{\partial%20o_1}{\partial%20a_{h_1}}%20\frac{\partial%20a_{h_1}}{\partial%20h_1}%20\frac{\partial{h_1}}{\partial{w_1}})
- The above is the Derivative of the Error wrt ![formula](https://render.githubusercontent.com/render/math?math=w_1). We similarly calculated the Partial Derivative wrt ![formula](https://render.githubusercontent.com/render/math?math=w_2), ![formula](https://render.githubusercontent.com/render/math?math=w_3) and ![formula](https://render.githubusercontent.com/render/math?math=w_4).
- Once all the weights were updated, then once again the cycle of forward propagtion and backward propagation repeats as many times as the number of epochs is (We ran for approximately 360 epochs).
## Learing Rate

- The rate at which updation of weights takes place is controlled by the learning rate ![formula](https://render.githubusercontent.com/render/math?math=\eta)
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%200.1) :
- [![E-total-0-1.png](https://i.postimg.cc/yNszW2FT/E-total-0-1.png)](https://postimg.cc/K1qWHqpK)
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%200.2) :
- [![E-total-0-2.png](https://i.postimg.cc/bvJtXVsm/E-total-0-2.png)](https://postimg.cc/WdB3gnmZ)
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%200.5) :
- [![E-total-0-5.png](https://i.postimg.cc/tJPr34bn/E-total-0-5.png)](https://postimg.cc/YLrfkkht)
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%200.8) :
- [![E-total-0-8.png](https://i.postimg.cc/ncF7YLn8/E-total-0-8.png)](https://postimg.cc/mhntTBcV)
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%201) :
- [![E-total-1-0.png](https://i.postimg.cc/gjYwDMFy/E-total-1-0.png)](https://postimg.cc/k6pXMv5D)
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%202) :
- [![E-total-2-0.png](https://i.postimg.cc/dtN72mWd/E-total-2-0.png)](https://postimg.cc/KRLvFLhv)
