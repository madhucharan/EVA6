# Backpropogation


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
- GRAPH
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%200.2) :
- GRAPH
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%200.5) :
- GRAPH
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%200.8) :
- GRAPH
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%201) :
- GRAPH
- Learning Rate ![formula](https://render.githubusercontent.com/render/math?math=\eta%20=%202) :
- GRAPH
